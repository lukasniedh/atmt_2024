import subprocess
import matplotlib.pyplot as plt
import json

beam_sizes = [5]

bleu_scores = []

data = "data/en-fr/prepared"
dicts = "data/en-fr/prepared"
checkpoint_path = "assignments/03/learning-rate-a/checkpoints/checkpoint_best.pt"

for beam_size in beam_sizes:
    print(f"Running translation with beam size {beam_size}...")
    
    # Run your translation script with the beam size
    translation_command = [
        "python", "translate_beam.py",
        "--data", data,
        "--dicts", dicts,
        "--checkpoint-path", checkpoint_path,
        "--output", f"assignments/05/beam_sizes/translation_{beam_size}.txt",
        "--beam-size", str(beam_size)
    ]
    subprocess.run(translation_command, check=True)
    
    # Run the post-processing bash script

    postprocess_command = [
        "bash", "scripts/postprocess.sh",
        f"assignments/05/beam_sizes/translation_{beam_size}.txt",
        f"assignments/05/beam_sizes/translation_{beam_size}.p.txt",
        "en"
    ]
    subprocess.run(postprocess_command, check=True)
    
    sacrebleu_command = f"sacrebleu data/en-fr/raw/test.en < assignments/05/beam_sizes/translation_{beam_size}.p.txt"

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


