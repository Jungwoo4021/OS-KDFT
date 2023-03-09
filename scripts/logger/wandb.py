import os
import torch
from .interface import ExperimentLogger

class WandbLogger(ExperimentLogger):
	"""Save experiment logs to wandb
	"""
	def __init__(self, path, name, group, project, entity, tags, script=None, save_dir = None):
		import wandb
		from wandb import AlertLevel
		self.wandb = wandb
		self.run = wandb.init(
				group=group,
				project=project,
				entity=entity,
				tags=tags
			)
		wandb.run.name = name
		path = os.path.join("/", path, project, "/".join(tags), name)
		self.paths = {
			'model' : f'{path}/model',
		}
		# upload zip file
		wandb.save(save_dir + "/script/script.zip")


	def log_metric(self, name, value, step=None):
		if step is not None:
			self.wandb.log({
				name: value,
				'step': step})
		else:
			self.wandb.log({name: value})

	def log_text(self, name, text):
		pass

	def log_image(self, name, image):
		self.wandb.log({name: [self.wandb.Image(image)]})

	def log_parameter(self, dictionary):
		self.wandb.config.update(dictionary)

	def save_model(self, name, state_dict):
		path = f'{self.paths["model"]}/BestModel.pt'
		self.wandb.save(path)

	def finish(self):
		self.wandb.finish()