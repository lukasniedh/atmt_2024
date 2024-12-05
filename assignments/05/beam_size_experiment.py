import subprocess
import matplotlib.pyplot as plt
import json
import os

beam_sizes = [1,3,5,10,15,20,25]

bleu_scores = []

data = "data/en-fr/prepared"
dicts = "data/en-fr/prepared"
checkpoint_path = "assignments/03/learning-rate-a/checkpoints/checkpoint_best.pt"

for beam_size in beam_sizes:

    # Run your translation script with the beam size (if it's not translated yet)
    translation_output_file = f"assignments/05/01_beam_sizes/translation_{beam_size}.txt"

    if not os.path.exists(translation_output_file):
        translation_command = [
            "python", "translate_beam.py",
            "--data", data,
            "--dicts", dicts,
            "--checkpoint-path", checkpoint_path,
            "--output", translation_output_file,
            "--beam-size", str(beam_size)
        ]
        subprocess.run(translation_command, check=True)
    
    # Run the post-processing bash script
    postprocess_output_file = f"assignments/05/01_beam_sizes/translation_{beam_size}.p.txt"

    if not os.path.exists(postprocess_output_file):
        postprocess_command = [
            "bash", "scripts/postprocess.sh",
            f"assignments/05/01_beam_sizes/translation_{beam_size}.txt",
            postprocess_output_file,
            "en"
        ]
        subprocess.run(postprocess_command, check=True)
    
    # calculate BLEU score
    sacrebleu_command = f"sacrebleu assignments/05/01_beam_sizes/translation_{beam_size}.p.txt < data/en-fr/raw/test.en"

    result = subprocess.run(
        sacrebleu_command, 
        shell=True, 
        stdout=subprocess.PIPE,
        text=True,
        check=True
    )

    bleu_output = json.loads(result.stdout)

    print(f"Beam size {beam_size} gives bleu score: {bleu_output['score']}")

    bleu_scores.append(bleu_output)

#plot BLEU score vs beam size
scores = [entry["score"] for entry in bleu_scores]
plt.figure(figsize=(8, 6))
plt.plot(beam_sizes, scores, marker="o", linestyle="-", label="BLEU Score")
plt.title("BLEU Scores")
plt.xlabel("Beam Size")
plt.ylabel("BLEU Score")
plt.xticks(beam_sizes)
plt.grid(True)
plt.legend()
plt.savefig("assignments/05/01_beam_sizes/bleu_score_vs_beam_size.png", format="png", dpi=300, bbox_inches="tight")

#plot BLEU score vs Brevity Penalty
bp_values = []
for result in bleu_scores:
    verbose_score = result.get("verbose_score", "")
    bp_start = verbose_score.find("BP = ") + len("BP = ")
    bp_end = verbose_score.find(" ", bp_start)
    bp_values.append(float(verbose_score[bp_start:bp_end]))



plt.figure(figsize=(8, 6))
plt.scatter(bp_values, scores, label="[Beam size]")
plt.title("Brevity Penalty")
plt.xlabel("Brevity Penalty (BP)")
plt.ylabel("BLEU Score")

for bp, score, label in zip(bp_values, scores, beam_sizes):
    plt.text(bp, score, f" [{label}] ")

plt.legend()
plt.savefig("assignments/05/01_beam_sizes/BP_values_vs_bleu_score.png", format="png", dpi=300, bbox_inches="tight")