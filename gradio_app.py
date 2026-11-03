"""Gradio version of the demo, used for the Hugging Face Space. Run with: python gradio_app.py"""

# On ZeroGPU, `spaces` has to be imported before torch. It only exists on Hugging Face.
try:
    import spaces
except ImportError:
    spaces = None

import gradio as gr

from symptom_search import SymptomMatcher, load_diseases
from symptom_search.__main__ import DISCLAIMER
from symptom_search.embeddings import Encoder

matcher = SymptomMatcher(load_diseases(), Encoder())

# The Space runs on ZeroGPU hardware, which refuses to start an app without a GPU
# function. Searches are fast enough on CPU, so this is never called.
if spaces is not None:

    @spaces.GPU
    def _unused_gpu_function() -> None:
        pass


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def search(text: str, top_k: float) -> str:
    if not text.strip():
        return ""
    matches = matcher.search(text, top_k=int(top_k))
    if not matches:
        return "No known symptoms recognised. Try describing what you feel in more detail."

    lines = []
    for position, match in enumerate(matches, 1):
        lines.append(f"### {position}. {match.disease}")
        if match.evidence:
            lines.append("| You said | Known symptom | Similarity |\n|---|---|---:|")
            lines += [f"| {_cell(item.clause)} | {_cell(item.symptom)} | {item.similarity:.2f} |" for item in match.evidence]
        else:
            lines.append("Matched on the overall description.")
        lines.append("")
    return "\n".join(lines)


with gr.Blocks(title="Symptom Search") as demo:
    gr.Markdown(
        "# Symptom Search\n"
        "Ranks 975 conditions by how well their listed symptoms match your description. "
        "[Code and benchmark](https://github.com/william5711/symptom-search)\n\n"
        f"*{DISCLAIMER}*"
    )
    text = gr.Textbox(
        label="Describe your symptoms in your own words",
        placeholder="I'm thirsty all the time and keep getting up at night to pee...",
        lines=3,
    )
    top_k = gr.Slider(3, 20, value=5, step=1, label="Results to show")
    button = gr.Button("Search", variant="primary")
    results = gr.Markdown()
    gr.Examples(
        [
            ["I'm thirsty all the time and pee constantly, no fever, and my vision is blurry"],
            ["I have a sore throat and a fever, and my neck glands are swollen"],
            ["My ankle is swollen and it hurts to walk after I twisted it"],
        ],
        inputs=text,
    )
    button.click(search, [text, top_k], results)
    text.submit(search, [text, top_k], results)

if __name__ == "__main__":
    demo.launch()
