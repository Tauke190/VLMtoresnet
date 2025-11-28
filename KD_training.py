import argparse
import sys
import os

def main():
	parser = argparse.ArgumentParser(description="Knowledge Distillation Training Script")
	parser.add_argument('--distillation', type=str, required=True, choices=[
		'contrastive', 'finalfeature', 'intermediate', 'masked_generative'],
		help='Type of distillation to run')
	parser.add_argument('--train_dir', type=str, required=False, help='Path to training directory')
	parser.add_argument('--val_dir', type=str, required=False, help='Path to validation directory')
	args = parser.parse_args()

	# Map distillation type to script
	distill_scripts = {
		'contrastive': 'distillation_techniques/contrastive_distillation_cliptoresnet.py',
		'finalfeature': 'distillation_techniques/finalfeature_distillation_cliptoresnet.py',
		'intermediate': 'distillation_techniques/intermediate_feature_distillation_cliptoresnet.py',
		'masked_generative': 'distillation_techniques/masked_generative_distillation.py',
	}

	script_path = distill_scripts[args.distillation]

	project_root = os.path.dirname(os.path.abspath(__file__))
	# Build command to run the selected script with the provided directories
	command = f'cd "{project_root}" && python {script_path} --train_dir {args.train_dir} --val_dir {args.val_dir}'
	print(f"Running: {command}")
	exit_code = os.system(command)
	if exit_code != 0:
		print(f"Error: Distillation script exited with code {exit_code}")
		sys.exit(exit_code)

if __name__ == "__main__":
	main()
