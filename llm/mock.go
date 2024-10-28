package llm

import (
	"context"
)

// Mock implements Client interface
type Mock struct{}

// MockProvider implements Provider interface
type MockProvider struct{}

// NewMockProvider creates a new instance of MockProvider
func NewMockProvider() *MockProvider {
	return &MockProvider{}
}

// NewClient returns a new Mock client
func (p *MockProvider) NewClient(modelName string, apiKey string) (Client, error) {
	return &Mock{}, nil
}

// Models returns the models available in Mock
func (p *MockProvider) Models() []string {
	return []string{"mock-model"}
}

// GenerateResponse returns a mock response
func (m *Mock) GenerateResponse(ctx context.Context, messages []Message) (string, error) {
	return "This is a mock response.", nil
}

/*
// Models returns the models available in Mock (for completeness)
func (m *Mock) Models() []string {
	return []string{"mock-model"}
}
*/
