# Decision Support Tool

A command-line tool that provides a decision support system using a filesystem-based decision tree, integrated with Large Language Models (LLMs) via OpenAI's API. The tool monitors a directory structure, processes user messages, handles attachments (like PDFs), and interacts with an LLM to generate responses, summaries, and metrics.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Daemon](#starting-the-daemon)
  - [Interacting with the Tool](#interacting-with-the-tool)
  - [Handling Attachments](#handling-attachments)
  - [Summarizing Paths](#summarizing-paths)
- [Directory Structure](#directory-structure)
- [Design and Architecture](#design-and-architecture)
  - [Filesystem-Based Decision Tree](#filesystem-based-decision-tree)
  - [File Monitoring with `fsnotify`](#file-monitoring-with-fsnotify)
  - [LLM Interaction](#llm-interaction)
  - [Attachments Handling](#attachments-handling)
  - [Metrics Tracking](#metrics-tracking)
- [Philosophy and Considerations](#philosophy-and-considerations)
- [Configuration](#configuration)
- [Security Considerations](#security-considerations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Filesystem-Based Decision Tree**: Represents conversations and decision paths as directories and files.
- **LLM Integration**: Interacts with OpenAI's API to process user messages and generate responses.
- **Attachment Handling**: Processes attachments like PDFs by extracting text content.
- **Summarization**: Summarizes conversation paths to manage context length.
- **Metrics Tracking**: Tracks user-defined and system-generated metrics for each decision node.
- **Editor Integration**: Designed to work seamlessly with text editors like Vim, Emacs, or VSCode.

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/decision_support_tool.git
   cd decision_support_tool
   ```

2. **Install Dependencies**:

   Ensure you have Go installed (version 1.16 or higher).

   ```bash
   go mod download
   ```

   Install required packages:

   ```bash
   go get github.com/fsnotify/fsnotify
   go get github.com/sashabaranov/go-openai
   go get rsc.io/pdf
   go get github.com/spf13/cobra
   ```

3. **Build the Application**:

   ```bash
   go build -o decision_tool main.go
   ```

---

## Usage

### Starting the Daemon

Run the tool by specifying the path to watch and your OpenAI API key:

```bash
./decision_tool --path /path/to/root --api-key YOUR_OPENAI_API_KEY
```

- **`--path`**: The root directory to monitor (default is the current directory).
- **`--api-key`**: Your OpenAI API key (required).

### Interacting with the Tool

1. **Open Your Editor**:

   - Navigate to the root directory specified in the `--path` argument.
   - Open `user_message.txt` in your preferred text editor.

2. **Compose a Message**:

   - Write your message or query in `user_message.txt`.
   - Save the file.

3. **Processing**:

   - The daemon detects the change and sends the message, along with the conversation context, to the LLM.
   - The response is written to `llm_response.txt` in the same directory.

4. **View the Response**:

   - Open `llm_response.txt` in your editor to view the LLM's reply.

5. **Creating New Decision Nodes**:

   - To explore different paths, create a new directory within the current one.
   - Use the provided function (or script) to generate a new decision node with a human-readable name.

### Handling Attachments

- **Adding an Attachment**:

  - Place a PDF file (e.g., `attachment.pdf`) in the current directory.
  - The daemon detects the PDF and extracts its text content.

- **Extracted Text**:

  - The extracted text is saved as `attachment.pdf.txt` alongside the original PDF.
  - This text can be included in API calls or referenced in messages.

### Summarizing Paths

- **Purpose**:

  - Summarizing a path helps manage the context length by condensing previous messages.

- **How to Summarize**:

  - Trigger the summarization function for the desired path.
  - A summary is generated and saved as `summary.txt` in that directory.

---

## Directory Structure

The tool organizes conversations and decisions into a hierarchical filesystem structure.

**Example**:

```
root/
├── user_message.txt
├── llm_response.txt
├── metrics.json
├── summary.txt
├── attachment.pdf
├── attachment.pdf.txt
├── decision1_explore_options_uuid/
│   ├── user_message.txt
│   ├── llm_response.txt
│   ├── metrics.json
│   └── summary.txt
└── decision2_alternative_path_uuid/
    ├── user_message.txt
    ├── llm_response.txt
    ├── metrics.json
    └── summary.txt
```

- **Directories**: Represent decision nodes or branches.
- **Files**:
  - `user_message.txt`: User's input at that node.
  - `llm_response.txt`: LLM's response.
  - `metrics.json`: Metrics related to that node.
  - `summary.txt`: Summary of the conversation up to that point.
  - Attachments: Any additional files like PDFs, along with their extracted text versions.

---

## Design and Architecture

### Filesystem-Based Decision Tree

- **Hierarchy Representation**: The conversation and decision paths are represented as directories and files, mirroring a decision tree.
- **Human-Readable Names**: Directories are named using a combination of descriptors and unique identifiers for clarity and uniqueness.
- **Flexibility**: Users can navigate, modify, and extend the tree using standard filesystem operations.

### File Monitoring with `fsnotify`

- **Real-Time Detection**: The tool uses `fsnotify` to watch for file changes in the directory tree.
- **Event Handling**: On detecting changes to `user_message.txt` or attachments, appropriate handlers are invoked.
- **Dynamic Monitoring**: As new directories are added, they are included in the watch list.

### LLM Interaction

- **Context Building**: The tool builds the conversation context by traversing from the root to the current node, collecting messages.
- **API Integration**: Interacts with OpenAI's API to send the context and receive responses.
- **Response Handling**: LLM responses are saved in the corresponding directory for user access.

### Attachments Handling

- **Supported Formats**: Currently supports PDFs; can be extended to other formats.
- **Text Extraction**: Uses the `rsc.io/pdf` package to extract text content from PDFs.
- **Usage**: Extracted text can be included in LLM prompts or analyzed separately.

### Metrics Tracking

- **Purpose**: Tracks user-defined metrics (e.g., performance scores) and system-generated metrics (e.g., completeness).
- **Storage**: Metrics are stored in `metrics.json` within each directory.
- **Updates**: Metrics are updated after processing messages or based on user input.

---

## Philosophy and Considerations

This tool is designed with simplicity, transparency, and user empowerment in mind.

### Simplicity and Accessibility

- **Filesystem as a Database**: Leveraging the native filesystem avoids the complexity of database setup and management.
- **Editor Integration**: By working within familiar text editors, users can seamlessly incorporate the tool into their workflows.

### Transparency and Control

- **Direct Data Access**: Users have direct access to all messages, responses, and attachments.
- **Modularity**: The hierarchical structure allows for easy modification and troubleshooting of specific conversation paths.

### Flexibility and Extensibility

- **Attachment Handling**: Storing files directly allows for diverse file types and easy extension to new formats.
- **Human-Readable Structure**: Combining descriptive names with unique identifiers balances clarity and uniqueness.

### Decision Support Enhancement

- **Structured Exploration**: Represents and manages LLM interactions as a decision tree, allowing users to explore multiple paths.
- **Metric Tracking**: Enables quantitative evaluation of options based on user-defined and system metrics.
- **Summarization and Merging**: Helps manage context length and combine insights from different paths.

### Considerations

- **Scalability**: While the filesystem approach is straightforward, it may face limitations with very large or deep directory structures.
- **Concurrency**: Managing simultaneous access by multiple users requires careful consideration to prevent conflicts.
- **Performance**: Monitoring and processing files in real-time necessitates efficient coding practices.

---

## Configuration

- **API Key**: Provide your OpenAI API key using the `--api-key` flag or set it as an environment variable.
- **Model Parameters**: Adjust model parameters like `maxTokens` and `temperature` in the source code as needed.
- **Watch Path**: Specify the root directory to monitor using the `--path` flag (default is the current directory).

---

## Security Considerations

- **API Key Handling**:

  - Do not hard-code the API key in the code.
  - Use command-line arguments or environment variables.
  - Protect the API key from unauthorized access.

- **Data Privacy**:

  - Be cautious with sensitive information included in messages or attachments.
  - Ensure that logs and outputs do not expose confidential data.

- **Error Logging**:

  - Avoid logging sensitive content.
  - Implement proper error handling to prevent leaks.

---

## Future Enhancements

- **Editor Plugins**: Develop plugins for Vim, Emacs, and VSCode to streamline interactions.
- **Visualization Tools**: Create scripts or integrations to visualize the decision tree within the editor.
- **Extended Attachment Support**: Add handlers for more file types like images or Word documents.
- **Concurrency Management**: Implement mechanisms to handle simultaneous users or integrate with version control systems.
- **Performance Optimization**: Enhance the efficiency of file monitoring and processing for larger projects.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Click on the 'Fork' button on the repository page.

2. **Create a Branch**: Create a new branch for your feature or bug fix.

   ```bash
   git checkout -b feature/your_feature_name
   ```

3. **Make Changes**: Implement your feature or fix.

4. **Commit Changes**:

   ```bash
   git commit -am 'Add new feature'
   ```

5. **Push to Branch**:

   ```bash
   git push origin feature/your_feature_name
   ```

6. **Create Pull Request**: Open a pull request on GitHub.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

**Note**: This README provides a foundational overview. Please refer to the code comments and documentation within the source files for more detailed explanations of functions and implementation specifics.
