//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using System.Text;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.ChatCompletion;
using dotenv.net;

#pragma warning disable SKEXP0001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
#pragma warning disable SKEXP0110 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

public class Program
{
    private const string WriterName = "CopyWriter";
    private const string WriterInstructions =
        """
        You are a copywriter with ten years of experience and are known for brevity and a dry humor.
        The goal is to refine and decide on the single best copy as an expert in the field.
        Only provide a single proposal per response.
        You're laser focused on the goal at hand.
        Don't waste time with chit chat.
        Consider suggestions when refining an idea.
        """;

    private const string ReviewerName = "ArtDirector";
    private const string ReviewerInstructions =
        """
        You are an art director who has opinions about copywriting born of a love for David Ogilvy.
        The goal is to determine if the given copy is acceptable to print.
        If so, state that it is approved.
        If not, provide insight on how to refine suggested copy without example.
        """;

    private const string IsChatDonePromptTemplate =
        """
        Determine if the copy has been approved.  If so, respond with a single word: yes

        History:
        {{$history}}
        """;

    private const string PickNextAgentPromptTemplate =
        $$$"""
        Your job is to determine which participant takes the next turn in a conversation according to the action of the most recent participant.
        State only the name of the participant to take the next turn.
        
        Choose only from these participants:
        - {{{ReviewerName}}}
        - {{{WriterName}}}
        
        Always follow these rules when selecting the next participant:
        - After user input, it is {{{WriterName}}}'a turn.
        - After {{{WriterName}}} replies, it is {{{ReviewerName}}}'s turn.
        - After {{{ReviewerName}}} provides feedback, it is {{{WriterName}}}'s turn.

        History:
        {{$history}}
        """;

    private static string GetEnvVariable(IDictionary<string, string> envVars, string key, string defaultValue = "")
    {
        string keyValue = envVars.TryGetValue(key, out string? value) && !string.IsNullOrEmpty(value)
            ? value
            : Environment.GetEnvironmentVariable(key) ?? defaultValue;

        if (string.IsNullOrEmpty(keyValue))
        {
            Console.WriteLine($"Environment variable {key} is not set.");
            Console.WriteLine(
                "\nYou can easily do that using the Azure AI CLI by doing one of the following:\n" +
                "\n  ai init\n  ai dev shell\n  dotnet run\n\n  or\n\n  ai init\n  ai dev shell --run \"dotnet run\""
            );
            throw new ArgumentException($"Environment variable {key} is not set.");
        }
        return keyValue;
    }


    public static async Task<int> Main(string[] args)
    {
        DotEnv.Load();
        var envVars = DotEnv.Read();

        string azureOpenAISystemPrompt = GetEnvVariable(envVars, "AZURE_OPENAI_SYSTEM_PROMPT", "You are a helpful AI assistant.");
        string AZURE_OPENAI_API_KEY = GetEnvVariable(envVars, "AZURE_OPENAI_API_KEY");
        string AZURE_OPENAI_ENDPOINT = GetEnvVariable(envVars, "AZURE_OPENAI_ENDPOINT");
        string AZURE_OPENAI_CHAT_DEPLOYMENT = GetEnvVariable(envVars, "AZURE_OPENAI_CHAT_DEPLOYMENT");

        var builder = Kernel.CreateBuilder();
        builder.AddAzureOpenAIChatCompletion(AZURE_OPENAI_CHAT_DEPLOYMENT!, AZURE_OPENAI_ENDPOINT!, AZURE_OPENAI_API_KEY!);

        var newKernel = new Func<Kernel>(() => builder.Build());

        // Loop until the user types 'exit'
        while (true)
        {
            // Get user input
            Console.Write("User: ");
            var userPrompt = Console.ReadLine();
            if (string.IsNullOrEmpty(userPrompt) || userPrompt == "exit") break;

            // Get the response
            Console.Write("\nAssistant: ");
            await GetChatMessageContentsAsync(userPrompt, newKernel);

            Console.WriteLine("\rAssistant: Done\n");
        }

        return 0;
    }

    private static async Task<string> GetChatMessageContentsAsync(string input, Func<Kernel> newKernel)
    {
        // The writer agent uses the WriterInstructions to guide the conversation.
        ChatCompletionAgent writer = new()
        {
            Instructions = WriterInstructions,
            Name = WriterName,
            Kernel = newKernel(),
        };

        // The reviewer agent uses the ReviewerInstructions to guide the conversation.
        ChatCompletionAgent reviewer = new()
        {
            Instructions = ReviewerInstructions,
            Name = ReviewerName,
            Kernel = newKernel(),
        };

        // Use a custom selection strategy to determine which agent should take the next turn, again using a KernelFunction prompt template.
        KernelFunction pickNextAgentFunction = KernelFunctionFactory.CreateFromPrompt(PickNextAgentPromptTemplate);
        KernelFunctionSelectionStrategy selectionStrategy = new KernelFunctionSelectionStrategy(pickNextAgentFunction, newKernel())
        {
            ResultParser = (result) => result.GetValue<string>() ?? WriterName,
            HistoryVariableName = "history", // The history variable is used to track the conversation history.
            AgentsVariableName = "agents",
        };

        // Use a custom termination strategy to determine when the chat is done, using a KernelFunction prompt template.
        KernelFunction isChatDoneFunction = KernelFunctionFactory.CreateFromPrompt(IsChatDonePromptTemplate);
        KernelFunctionTerminationStrategy terminationStrategy = new KernelFunctionTerminationStrategy(isChatDoneFunction, newKernel())
        {
            Agents = [reviewer],
            ResultParser = (result) => result.GetValue<string>()?.Contains("yes", StringComparison.OrdinalIgnoreCase) ?? false,
            HistoryVariableName = "history", // The history variable is used to track the conversation history.
            MaximumIterations = 10, // Limit the number of iterations to prevent the conversation from going on indefinitely.
        };

        // Create a group chat with the writer and reviewer agents, and the custom termination and selection strategies.
        var agents = new[] { writer, reviewer };
        AgentGroupChat chat = new(agents)
        {
            ExecutionSettings = new()
            {
                SelectionStrategy = selectionStrategy,
                TerminationStrategy = terminationStrategy
            }
        };

        // Start the chat by adding the initial user input
        chat.AddChatMessage(new ChatMessageContent(AuthorRole.User, input));

        // Process the chat messages calling the callback for each message
        var responseContent = new StringBuilder();
        await foreach (var content in chat.InvokeAsync())
        {
            var hasAuthor = !string.IsNullOrEmpty(content.AuthorName);
            var output = hasAuthor
                ? $"{content.Role}-{content.AuthorName}: {content.Content}\n"
                : $"{content.Role}: {content.Content}\n";

            if (content.AuthorName == WriterName)
            {
                Console.ForegroundColor = ConsoleColor.Magenta;  // Set text color to blue for CopyWriter
            }
            else if (content.AuthorName == ReviewerName)
            {
                Console.ForegroundColor = ConsoleColor.Green;  // Set text color to green for ArtDirector
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;  // Set text color to dark gray for others
            }

            Console.WriteLine($"\r{output}");
            responseContent.AppendLine(output);

            Console.ForegroundColor = ConsoleColor.White;  // Reset to default color
            Console.Write("Assistant: ");
        }

        return responseContent.ToString();
    }
}
