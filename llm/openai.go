package llm

import (
	"context"
	"errors"
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
	Model       string                   `json:"model"`
	Messages    []ChatCompletionMessage  `json:"messages"`
	MaxTokens   int                      `json:"max_tokens"`
	Temperature float32                  `json:"temperature"`
}

// ChatCompletionResponse struct represents a response from the OpenAI chat completion API
type ChatCompletionResponse struct {
	Choices []ChatCompletionChoice `json:"choices"`
}

// ChatCompletionChoice struct represents a choice in the chat completion response
type ChatCompletionChoice struct {
	Message ChatCompletionMessage `json:"message"`
}

// Client is the interface that wraps the CreateChatCompletion method
type Client interface {
	CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
}

// openAIClient implements Client interface
type openAIClient struct {
	apiKey string
}

// NewClient creates a new instance of OpenAI client
func NewClient(apiKey string) Client {
	return &openAIClient{apiKey: apiKey}
}

// CreateChatCompletion sends a request to the OpenAI API and returns a completion response
func (c *openAIClient) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	// Mocked response for demonstration purposes
	if c.apiKey == "" {
		return ChatCompletionResponse{}, errors.New("API key is not set")
	}

	response := ChatCompletionResponse{
		Choices: []ChatCompletionChoice{
			{Message: ChatCompletionMessage{Role: ChatMessageRoleAssistant, Content: "This is a placeholder response."}},
		},
	}

	return response, nil
}
