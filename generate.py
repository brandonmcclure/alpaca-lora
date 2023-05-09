import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import logging

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    offload_folder: str = '/root/offload'
):

    # set up logger to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    app_title="🦙🌲 Alpaca-LoRA"
    app_description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora)."  # noqa: E501    
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    logger.info(f'{app_title} started')
    prompter = Prompter(prompt_template)
    logger.info(f'Getting LlamaTokenizer from the base model: {base_model}')
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    logger.info(f'tokenizer: {tokenizer}')

    

    logger.debug(f'Detected device: {device}')

    
    if device == "cuda":
        logger.debug(f'Using cuda device models')

        logger.debug(f'Determining memory settings')
        free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
        logger.debug(f'mem: {torch.cuda.mem_get_info()}')
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)}GB'
        logger.debug(f'max_memory: {max_memory}')
        n_gpus = torch.cuda.device_count()
        logger.debug(f'n_gpus: {n_gpus}')
        max_memory = {i: max_memory for i in range(n_gpus)}
        logger.debug(f'max_memory: {max_memory}')

        quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

        logger.debug(f'offload_folder: {offload_folder}')
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=quantization_config,
            offload_folder=offload_folder
        )
        logger.debug(f'model after LlamaForCausalLM: {model}')
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            load_in_8bit=load_8bit,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=quantization_config,
            offload_folder=offload_folder
        )
        logger.info(f'model after Peft: {model}')
    elif device == "mps":
        logger.debug(f'Using mps device models')
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        logger.debug(f'model after LlamaForCausalLM: {model}')
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        logger.info(f'model after Peft: {model}')
    else:
        logger.debug(f'other device "{device}. Using the default models')
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        logger.debug(f'model after LlamaForCausalLM: {model}')
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
        logger.info(f'model after Peft: {model}')

    logger.info('unwind broken decapoda-research config')
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    logger.debug('model.eval()')
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        logger.info(f'Evaluating')
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        logger.info(f'generation_config: {generation_config}')

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        logger.info(f'generate_params: {generate_params}')
        logger.info(f'stream_output: {stream_output}')
        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            logger.info(f'Torching with no_grad!')
            logger.info(f'generation_config: {generation_config}')
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    logger.info('Launch Gradio interface')
    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
