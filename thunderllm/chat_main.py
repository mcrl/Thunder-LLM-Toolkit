import os
import sys

import thunderllm
import thunderllm.generation

import thunderllm.chat_arguments as chat_arguments


def cli_chat():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = chat_arguments.parse_args()

    process_grid = thunderllm.distributed.initialize_process_grid()
    device = thunderllm.distributed.configure_device(args)

    if args.model_arch is None and args.hf_model_path is None:
        raise ValueError(
            "Please provide either --model-arch or --hf-model-path")

    generator = thunderllm.generation.get_generator(args)

    while True:
        print("Press Q to quit")
        prompt = "User: "
        cmd = input(prompt)
        if cmd == "Q":
            prompt = "Are you sure you want to quit? (y/N): "
            if input(prompt).strip() == "y":
                break
        generated = generator.generate_sentence(cmd)
        if args.debug:
            print("\n")
        print("Generated:", generated)


if __name__ == "__main__":
    cli_chat()
