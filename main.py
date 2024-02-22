import src.llm.bedrock as bedrock
import src.ui.gradio.chatbot as chatbot

boto3Bedrock = bedrock.Boto3Bedrock(model_id="anthropic.claude-v2")
chatbot_UI = chatbot.GradioUI(boto3_bedrock=boto3Bedrock)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chatbot_UI.launch()
