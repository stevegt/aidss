package llm

import (
	"context"
)

// Mock implements Client interface
type Mock struct {
	model Model
}

// MockProvider implements Provider interface
type MockProvider struct{}

// Mock model
var mockModel = Model{
	Name:        "mock-model",
	MaxTokens:   1000,
	Temperature: 0.7,
}

// NewMockProvider creates a new instance of MockProvider
func NewMockProvider() *MockProvider {
	return &MockProvider{}
}

// NewClient returns a new Mock client
func (p *MockProvider) NewClient(modelName string) (Client, error) {
	return &Mock{
		model: mockModel,
	}, nil
}

// Models returns the models available in Mock
func (p *MockProvider) Models() []string {
	return []string{mockModel.Name}
}

// GenerateResponse returns a mock response
func (m *Mock) GenerateResponse(ctx context.Context, messages []Message) (string, error) {
	return "This is a mock response.", nil
}
