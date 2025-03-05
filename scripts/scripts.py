import os
import json
import numpy as np
import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks

script_directory = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(script_directory, "combination-lora.css")
with open(css_path, "r", encoding="utf-8") as f:
    css_content = f.read()

def update_action_buttons(mode):
    if mode == "prompt":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def update_wildcard_input(mode):
    if mode == "wildcard":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def process_from_ui(lora_path, select_mode, wildcard_name, weight_min, weight_max):
    return process_lora_files(lora_path, select_mode, wildcard_name, weight_min, weight_max)

def generate_ui_output(lora_path, select_mode, wildcard_name, weight_min, weight_max):
    if select_mode == "prompt":
        if not lora_path.strip():
            return "Error: LoRA Path is required.\nエラー：loraのフォルダー指定は必須です"
        result = f"LoRA Path: {lora_path}\nSelect Mode: {select_mode}\nLoRA Weight: {weight_min} - {weight_max}"
    else:
        if not lora_path.strip() or not wildcard_name.strip():
            return "Error: Both LoRA Path and Wildcard Name are required in wildcard mode.\nエラー：ワイルドカードモードでは、ファイルネームが必要です"
        result = f"LoRA Path: {lora_path}\nWildcard Name: {wildcard_name}\nSelect Mode: {select_mode}\nLoRA Weight: {weight_min} - {weight_max}"
    return result

def process_lora_files(lora_folder, output_type, wildcard_file_name, ui_weight_min, ui_weight_max):
    target_extensions = ['.safetensors', '.ckpt']
    if not lora_folder.strip():
        return "Error: LoRA Path is required.\nエラー：loraのフォルダー指定は必須です"
    if output_type == "prompt":
        valid_files = [f for f in os.listdir(lora_folder) if any(f.lower().endswith(ext) for ext in target_extensions)]
        lora_combination_min = "1"
        lora_combination_max = str(len(valid_files))
        lora_codes = []
        for filename in valid_files:
            basename = os.path.splitext(filename)[0]
            json_path = os.path.join(lora_folder, f"{basename}.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(lora_folder, f"{filename}.json")
            minimum_value = str(ui_weight_min)
            maximum_value = str(ui_weight_max)
            activation_text = ""
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        minimum_value = str(data.get("minimum_value", minimum_value))
                        maximum_value = str(data.get("maximum_value", maximum_value))
                        activation_text = data.get("activation tex", data.get("activation text", data.get("activation_text", "")))
                except Exception as e:
                    print(f"JSONの読み込みエラー({filename}): {e}")
            min_val = float(minimum_value)
            max_val = float(maximum_value)
            weight_values = np.arange(min_val, max_val + 0.1, 0.1).round(1)
            weights = "|".join(map(str, weight_values))
            if activation_text:
                lora_code = f"<lora:{basename}:{{{weights}}}>{activation_text}"
            else:
                lora_code = f"<lora:{basename}:{{{weights}}}>"
            lora_codes.append(lora_code)
        prompt_output = "{" + f"{lora_combination_min}-{lora_combination_max}$${'|'.join(lora_codes)}" + "}"
        return prompt_output
    elif output_type == "wildcard":
        if not wildcard_file_name.strip():
            return "Error: Both LoRA Path and Wildcard Name are required in wildcard mode.\nエラー：ワイルドカードモードでは、ファイルネームが必要です"
        valid_files = [f for f in os.listdir(lora_folder) if any(f.endswith(ext) for ext in target_extensions)]
        file_count = len(valid_files)
        lora_list_items = []
        for filename in valid_files:
            basename = os.path.splitext(filename)[0]
            default_lora_name = basename
            activation_text = ""
            minimum_value = str(ui_weight_min)
            maximum_value = str(ui_weight_max)
            json_path = os.path.join(lora_folder, f"{basename}.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(lora_folder, f"{filename}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        activation_text = data.get("activation tex", data.get("activation text", data.get("activation_text", "")))
                        minimum_value = str(data.get("minimum_value", minimum_value))
                        maximum_value = str(data.get("maximum_value", maximum_value))
                except Exception as e:
                    print(f"JSONの読み込みエラー({filename}): {e}")
            min_val = float(minimum_value)
            max_val = float(maximum_value)
            weight_values = np.arange(min_val, max_val + 0.1, 0.1).round(1)
            weights = "|".join(map(str, weight_values))
            entry = f"- <lora:{default_lora_name}:{{{weights}}}>"
            if activation_text:
                entry += f" {activation_text}"
            lora_list_items.append(entry)
        combination_patterns = []
        for i in range(1, file_count + 1):
            pattern = "{" + str(i) + "$$" + "|".join(["__Lora-list__"] * i) + "}"
            combination_patterns.append(pattern)
        yaml_content = f"{wildcard_file_name}:\n"
        yaml_content += wildcard_file_name- + "lora-combination:\n"
        for pat in combination_patterns:
            yaml_content += "    - >-\n      " + pat + "\n"
        yaml_content += wildcard_file_name- + "lora-list:\n"
        for item in lora_list_items:
            yaml_content += "    " + item + "\n"
        extensions_dir = os.path.dirname(os.path.dirname(script_directory))
        dynamic_prompts_dir = os.path.join(extensions_dir, "sd-dynamic-prompts", "wildcards")
        if not os.path.exists(dynamic_prompts_dir):
            return "Error: dynamic promptsが必要です"
        os.makedirs(dynamic_prompts_dir, exist_ok=True)
        yaml_file_path = os.path.join(dynamic_prompts_dir, f"{wildcard_file_name}.yaml")
        with open(yaml_file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        output_message = "created a wildcard \n ワイルドカードを作成しました。 " + yaml_file_path + "\n" 
        return output_message

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, css=css_content) as ui_component:
        gr.HTML(f"<style>{css_content}</style>")
        with gr.Row():
            with gr.Column():
                lora_path = gr.Textbox(label="LoRA Path", placeholder="Enter the path to your LoRA model", lines=1, elem_id="lora_path")
                select_path = gr.Radio(choices=["wildcard", "prompt"], label="Select Mode", value="wildcard", elem_id="select_mode")
                wildcard_name = gr.Textbox(label="Wildcard name input,The same name will be overwritten", placeholder="Enter wildcard name", lines=1, elem_id="wildcard_name", visible=True)
                with gr.Row():
                    weight_min = gr.Slider(label="LoRA weight min", minimum=0.1, maximum=1, value=0.1, step=0.1, elem_id="weight_min")
                    weight_max = gr.Slider(label="LoRA weight max", minimum=0.1, maximum=1, value=1, step=0.1, elem_id="weight_max")
                generate_button = gr.Button("Generate")
            with gr.Column():
                with gr.Row():
                    output_text = gr.Textbox(label="", interactive=False, elem_id="result_box")
                with gr.Row():
                    copy_button = gr.Button("Copy", elem_id="copy_button", visible=False)
        select_path.change(fn=update_action_buttons, inputs=select_path, outputs=copy_button)
        select_path.change(fn=update_wildcard_input, inputs=select_path, outputs=wildcard_name)
        generate_button.click(fn=process_from_ui, inputs=[lora_path, select_path, wildcard_name, weight_min, weight_max], outputs=output_text)
        copy_button.click(fn=lambda x: x, inputs=output_text, outputs=output_text, _js="(text) => { navigator.clipboard.writeText(text); return text; }")
        return [(ui_component, "Combination lora", "combination-lora_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
