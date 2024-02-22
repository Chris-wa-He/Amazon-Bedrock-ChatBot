import gradio as gr

import src.llm.bedrock as bedrock


class GradioUI:
    def __init__(self, boto3_bedrock):
        self.history = []
        self.ui = self.create_ui(boto3_bedrock)

    def create_ui(self, boto3Bedrock):
        def predict(message, history):
            history = history or []
            response = boto3Bedrock.get_conversation().predict(input=message)
            history.append((message, response))
            return response

        return gr.ChatInterface(
            predict,
            chatbot=gr.Chatbot(height=300),
            textbox=gr.Textbox(placeholder="Ask me questions", container=False, scale=7),
            title="Amazon Bedrock",
            description="GUI for Amazon Bedrock",
            # theme="soft",
            # examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
            # cache_examples=True,
            retry_btn=None,
            undo_btn="Delete Previous",
            clear_btn="Clear",
        )

    def launch(self):
        self.ui.launch()

# boto3Bedrock = bedrock.Boto3Bedrock(model_id="anthropic.claude-v2")
# GradioUI(model_id="anthropic.claude-v2", boto3Bedrock=boto3Bedrock).launch()
