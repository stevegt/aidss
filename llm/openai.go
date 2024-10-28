package llm

import (
	"context"
	"errors"
	"os"
)

// Define constants for models
const (
	GPT4o     = "gpt-4o"
	O1Mini    = "o1-mini"
	O1Preview = "o1-preview"
)

// ChatMessageRole defines roles for chat messages
const (
	ChatMessageRoleUser      = "user"
	ChatMessageRoleAssistant = "assistant"
	ChatMessageRoleSystem    = "system"
)

// ChatCompletionMessage struct represents a message in the chat
type ChatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest struct represents a request to the OpenAI chat completion API
type ChatCompletionRequest struct {
	Model       string                  `json:"model"`
	Messages    []ChatCompletionMessage `json:"messages"`
	MaxTokens   int                     `json:"max_tokens"`
	Temperature float32                 `json:"temperature"`
}

// ChatCompletionResponse struct represents a response from the OpenAI chat completion API
type ChatCompletionResponse struct {
	Choices []ChatCompletionChoice `json:"choices"`
}

// ChatCompletionChoice struct represents a choice in the chat completion response
type ChatCompletionChoice struct {
	Message ChatCompletionMessage `json:"message"`
}

// Model struct represents a language model with its attributes
type Model struct {
	Name        string
	MaxTokens   int
	Temperature float32
}

// Map of model names to Model structs
var openAIModels = map[string]Model{
	GPT4o: {
		Name:        GPT4o,
		MaxTokens:   128000,
		Temperature: 0.7,
	},
	O1Mini: {
		Name:        O1Mini,
		MaxTokens:   128000,
		Temperature: 0.7,
	},
	O1Preview: {
		Name:        O1Preview,
		MaxTokens:   128000,
		Temperature: 0.7,
	},
}

// OpenAI implements Client interface
type OpenAI struct {
	apiKey string
	model  Model
}

// OpenAIProvider implements Provider interface
type OpenAIProvider struct {
	apiKey string
}

// NewOpenAIProvider creates a new instance of OpenAIProvider
func NewOpenAIProvider() *OpenAIProvider {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		// Return nil if API key is not set
		return nil
	}

	return &OpenAIProvider{
		apiKey: apiKey,
	}
}

// NewClient returns a new OpenAI client for the given model
func (p *OpenAIProvider) NewClient(modelName string) (Client, error) {
	model, ok := openAIModels[modelName]
	if !ok {
		return nil, errors.New("unsupported model: " + modelName)
	}

	return &OpenAI{
		apiKey: p.apiKey,
		model:  model,
	}, nil
}

// Models returns the models available in OpenAI
func (p *OpenAIProvider) Models() []string {
	models := make([]string, 0, len(openAIModels))
	for modelName := range openAIModels {
		models = append(models, modelName)
	}
	return models
}

// CreateChatCompletion sends a request to the OpenAI API and returns a completion response
func (o *OpenAI) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	// Mocked response for demonstration purposes
	if o.apiKey == "" {
		return ChatCompletionResponse{}, errors.New("API key is not set")
	}

	response := ChatCompletionResponse{
		Choices: []ChatCompletionChoice{
			{Message: ChatCompletionMessage{Role: ChatMessageRoleAssistant, Content: "This is a placeholder response from OpenAI."}},
		},
	}

	return response, nil
}

// GenerateResponse implements the Client interface
func (o *OpenAI) GenerateResponse(ctx context.Context, messages []Message) (string, error) {
	// Convert Messages to ChatCompletionMessages
	var chatMessages []ChatCompletionMessage
	for _, msg := range messages {
		chatMessages = append(chatMessages, ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	req := ChatCompletionRequest{
		Model:       o.model.Name,
		Messages:    chatMessages,
		MaxTokens:   o.model.MaxTokens,
		Temperature: o.model.Temperature,
	}

	resp, err := o.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Choices[0].Message.Content, nil
}
