import boto3
from botocore.exceptions import ClientError

from langchain.schema import BaseOutputParser, OutputParserException
from typing import Any, Dict, List, Optional, Type, cast
from langchain.output_parsers.json import parse_and_check_json_markdown
import json


###########################################################
################# Validate Inference Parameters ###########
###########################################################

titan_models_inf_param = {
    "temperature": "(float) Temperature",
    "topP": "(float) Top P",
    "maxTokenCount": "(int) Response Length",
    "stopSequences": "([string]) Stop Sequences",
}


claude_models_inf_param = {
    "temperature": "(float) Temperature",
    "topP": "(float) Top P",
    "topK": "(float) Top K",
    "max_tokens_to_sample": "(int) Maximum length",
    "stop_sequences": "([string]) Stop sequences",
}



jurassic_models_inf_param = {
    "temperature": "(float) Temperature",
    "topP": "(float) Top P",
    "maxTokens": "(int) Maximum completion length",
    "stopSequences": "([string]) Stop sequences",
    "presencePenalty": "(float) Presence penalty",
    "countPenalty": "(int) Count penalty",
    "frequencyPenalty": "(int) Frequency penalty",
    "applyToWhitespaces": "(bool) Whitespaces penalty",
    "applyToPunctuation": "(bool) Punctuations penalty",
    "applyToNumbers": "(bool) Numbers penalty",
    "applyToStopwords": "(bool) Stop words penalty",
    "applyToEmojis" : "(bool) Emojis penalty",
    
}



command_models_inf_param = {
    "temperature": "(float) Temperature",
    "p": "(float) Top P",
    "k": "(float) Top K",
    "return_likelihoods" : "(string) Return likelihoods - GENERATION, ALL, NONE ",
    "stream": "(bool) Stream",
    "max_tokens": "(int) Maximum length",
    "stop_sequences": "(str) Stop sequences",
    "num_generations": "(int) Number of generations"
}

stability_models_inf_param = {
    "cfg_scale": "(float) Prompt Strength",
    "steps": "(int) Seed",
}

mixtral_models_inf_param = {
    "max_tokens" : "(int) Maximum completion length",
    "stop" : "(str) Stop sequences",    
    "temperature": "(float) Temperature",
    "top_p": "(float) Top P",
    "top_k": "(float) Top K"
}


def validate_inference_parameters(model_id, inference_parameters):
    # Titan models
    if "titan" in model_id:
        selection = titan_models_inf_param
    elif "ai21" in model_id:
        selection = jurassic_models_inf_param
    elif "claude" in model_id:
        selection = claude_models_inf_param
    elif "command" in model_id:
        selection = command_models_inf_param
    elif "stability" in model_id:
        selection = stability_models_inf_param
    elif "mixtral" in model_id:
        selection = mixtral_models_inf_param
    else:
        raise ValueError("Check the model id. \
        Currently we only support the following family of models:\
        Amazon - Titan, AI21 Labs - Jurassic, Anthropic - Claude, \
        Cohere - Command and Stability AI - Stable Diffusion")

    for key in inference_parameters:
        if key not in selection:
            raise ValueError("'{}' is not a valid inference paramater for model: {}".format(key, model_id))

    return True


def validate_model_access(bedrock_runtime, model_id):
    """Validates access to requested model.
    
    Return True when model is accessible and False otherwise
    """
    try:
        if "titan" in model_id:
            request = {
                "inputText": "How are you?",
                "textGenerationConfig": {
                    "maxTokenCount": 64,
                    "temperature": 0.5,
                },
            }
        elif "mixtral" in model_id:
            request = {
                "prompt": "<s>[INST] How are you? [/INST]",
                "max_tokens": 64,
                "temperature": 0.5,
            }
        elif "claude" in model_id:
            # Updated Claude format matching the API structure
            request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 64,
                "temperature": 0.5,
                "system": "You are a helpful assistant.",
                "messages": [
                    {
                        "role": "user",
                        "content": "How are you?"
                    }
                ]
            }
        elif "command" in model_id:
            request = {
                "prompt": "How are you?",
                "max_tokens": 64,
                "temperature": 0.5,
            }
        elif "stability" in model_id:
            request = {
                "text_prompts": [{
                    "text": "How are you?",
                    "cfg_scale": 10,
                    "steps": 1,
                }]
            }
        else:
            raise ValueError("Check the model id. \
            Currently we only support the following family of models:\
            Amazon - Titan, AI21 Labs - Jurassic, Anthropic - Claude, \
            Cohere - Command and Stability AI - Stable Diffusion")

        bedrock_runtime.invoke_model(modelId=model_id, body=json.dumps(request))
        return True
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
            return False
        else:
            raise error
    

def validate_models_access(model_ids):
    """Validates access to list of model ids.

    Returns an empty list if all models are accessible and a list of inaccessible models otherwise.
    """
    bedrock_runtime = boto3.client(service_name="bedrock-runtime")

    return [model_id for model_id in model_ids if not validate_model_access(bedrock_runtime, model_id)]

