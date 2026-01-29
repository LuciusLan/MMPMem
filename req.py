import json
import random
import base64
import time

from openai import OpenAI

client = OpenAI(api_key="")

random.seed(1234)

GT = False

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

with open('subset.jsonl') as f:
    dataset = [json.loads(e) for e in f.readlines()]


template_gt = "Please answer the question regarding a visual feature of an organism (animal, plant, etc.). You will be provided with with a image regarding that organism, it is likely to contain the key information for answering the question. Please always output the final answer wrapped in box: \"\\boxed{answer text}\"\n\nQuestion: "

template_zs = "Please answer the question regarding a visual feature of an organism (animal, plant, etc.). Please always output the final answer wrapped in box: \"\\boxed{answer text}\"\n\nQuestion: "


response_ids = []
for ii, row in enumerate(dataset):
    gts = []
    for img, lab in row['images'].items():
        if lab == 1:
            gts.append(img)
    
    # select_gt = random.choice(gts)
    # select_gt = "/mnt/d/Dev/test_img/" + select_gt
    # gt_b64 = encode_image(select_gt)
    # gt_b64 = f"data:image/jpeg;base64,{gt_b64}"

    if GT:
        request = [{
            "role": "user",
            "content": [
                {
                    "type": "input_text", "text": template_gt+row['question'],
                },
                {
                    "type": "input_image",
                    "image_url": gt_b64,
                }
            ]
        }]
    else:
        request = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text", "text": template_zs+row['question']
                    }
                ]
            }
        ]
    response = client.responses.create(
        model="gpt-4o",
        input=request,

        # background=True,
        # service_tier="flex",
        # tools=[{"type": "web_search"}],
        # tool_choice="auto",
        # reasoning={"effort": "medium", "summary": "concise"},
        # max_tool_calls=5,
    )

    # while response.status in {"queued", "in_progress"}:
    #     print(f"Current status: {response.status}")
    #     time.sleep(10)
    #     response = client.responses.retrieve(response.id)
    #     if response.status == 'completed':
    #         print("completed")
    #         break

    # response_final = response
    # reasoning_path = []
    # for out_item in response_final.output:
    #     if type(out_item).__name__ == "ResponseReasoningItem":
    #         reasoning_path.append(["reason", [e.text for e in out_item.summary]])
    #     elif type(out_item).__name__ == "ResponseFunctionWebSearch":
    #         try:
    #             reasoning_path.append(["search",out_item.action.queries])
    #         except:
    #             pass
    #     elif type(out_item).__name__ == "ResponseOutputMessage":
    #         reasoning_path.append(["final", [e.text for e in out_item.content]])
    # response_ids.append({"qid": ii, "reasoning": reasoning_path})

    response_ids.append({"qid": ii, "pred": response.output_text})

with open('subset_answers/4o_zs.jsonl', 'w') as f:
    for row in response_ids:
        f.write(json.dumps(row))
        f.write('\n')



pass
