import os
from typing import Optional

# External Dependencies:
import boto3
from botocore.config import Config

from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory


class Boto3Bedrock:

    def __init__(self, model_id="anthropic.claude-v2"):
        self.model_id = model_id
        self.boto3_bedrock = Boto3Bedrock.get_bedrock_client(
            assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
            region=os.environ.get("AWS_DEFAULT_REGION", None),
            runtime=True
        )

        # modelId = "anthropic.claude-v2"
        self.cl_llm = Bedrock(
            model_id=self.model_id,
            client=self.boto3_bedrock,
            model_kwargs={"max_tokens_to_sample": 1000},
        )

        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.cl_llm, verbose=True, memory=self.memory
        )

    @staticmethod
    def get_bedrock_client(
            assumed_role: Optional[str] = None,
            region: Optional[str] = None,
            runtime: Optional[bool] = True,
    ):
        """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

        Parameters
        ----------
        assumed_role :
            Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
            specified, the current active credentials will be used.
        region :
            Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
            If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
        runtime :
            Optional choice of getting different client to perform operations with the Amazon Bedrock service.
        """
        if region is None:
            target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
        else:
            target_region = region

        print(f"Create new client\n  Using region: {target_region}")
        session_kwargs = {"region_name": target_region}
        client_kwargs = {**session_kwargs}

        profile_name = os.environ.get("AWS_PROFILE")
        if profile_name:
            print(f"  Using profile: {profile_name}")
            session_kwargs["profile_name"] = profile_name

        retry_config = Config(
            region_name=target_region,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )
        session = boto3.Session(**session_kwargs)

        if assumed_role:
            print(f"  Using role: {assumed_role}", end='')
            sts = session.client("sts")
            response = sts.assume_role(
                RoleArn=str(assumed_role),
                RoleSessionName="langchain-llm-1"
            )
            print(" ... successful!")
            client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
            client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
            client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

        if runtime:
            service_name = 'bedrock-runtime'
        else:
            service_name = 'bedrock'

        bedrock_client = session.client(
            service_name=service_name,
            config=retry_config,
            **client_kwargs
        )

        print("boto3 Bedrock client successfully created!")
        print(bedrock_client._endpoint)
        return bedrock_client

    def get_conversation(self):
        return self.conversation

# print(Boto3Bedrock().get_conversation().predict(input="Hello!"))
# print(conversation.predict(input="who are you?"))
