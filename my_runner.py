# from run_open_LLM import main
from run_mplight import main

import argparse

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--memo", type=str, default='LLMTLCSRun')
    # # parser.add_argument("--llm_model", type=str, default="lightgpt0")
    # parser.add_argument("--llm_model", type=str, default="llama-2-13b-chat-hf")
    # parser.add_argument("--llm_path", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    # # parser.add_argument("--llm_path", type=str, default="USAIL-HKUSTGZ/LLMLight-LightGPT")
    # parser.add_argument("--num_rounds", type=int, default=1)
    # parser.add_argument("--new_max_tokens", type=int, default=1024)
    # parser.add_argument("--proj_name", type=str, default="lightgpt_test_0")
    # parser.add_argument("--eightphase", action="store_true", default=False)
    # parser.add_argument("--multi_process", action="store_true", default=True)
    # parser.add_argument("--workers", type=int, default=1)
    # parser.add_argument("--dataset", type=str, default="jinan")
    # parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")

    parser.add_argument("--memo", type=str, default='MPLight')
    parser.add_argument("--mod", type=str, default="EfficientMPLight")
    parser.add_argument("--model", type=str, default="MPLight")
    parser.add_argument("--proj_name", type=str, default="mplight_test_4")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--multi_process", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")

    args = parser.parse_args()

    main(args)