from functools import partial
from vit_prisma.circuit.graph import Graph
from vit_prisma.circuit.attribute import attribute
from vit_prisma.circuit.attribute_node import attribute_node
from vit_prisma.circuit.metrics import get_metric
from vit_prisma.circuit.evaluate import evaluate_baseline, evaluate_graph, evaluate_area_under_curve

class CircuitAnalyzer:
    AVAILABLE_METHODS = ['exact', 'EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'UGS', 'information-flow-routes', 'random']
    def __init__(
        self,
        model,
        task: str,
        method: str,
        metric_name: str,
        level: str = 'edge',
        ablation: str = 'patching',
        ig_steps: int = 5,
        optimal_ablation_path: str = None,
        device: str = 'cuda',
    ):
        self.model = model
        self.task = task
        self.method = method
        self.level = level
        self.metric_name = metric_name
        self.ablation = ablation
        self.ig_steps = ig_steps
        self.optimal_ablation_path = optimal_ablation_path
        self.device = device

        self.graph = Graph.from_model(model)
        self.metric = get_metric(metric_name, task, model, model)
        self.attribution_metric = partial(self.metric, mean=True, loss=True)

    def run_analysis(
        self,
        clean_dataloader,
        intervention_dataloader=None,
    ):
        if self.level == 'edge':
            per_example_scores = attribute(
                self.model, self.graph,
                clean_dataloader,
                self.attribution_metric,
                self.method,
                self.ablation,
                intervention_dataloader=intervention_dataloader,
                ig_steps=self.ig_steps,
                optimal_ablation_path=self.optimal_ablation_path,
                device=self.device,
                task=self.task,
                model_name=self.model.cfg.model_name,
            )
            return self.graph, per_example_scores
        else:
            attribute_node(
                self.model, self.graph,
                clean_dataloader,
                self.attribution_metric,
                self.method,
                self.ablation,
                neuron=self.level == 'neuron',
                ig_steps=self.ig_steps,
                optimal_ablation_path=self.optimal_ablation_path,
            )
            return self.graph, None

    def run_evaluation(self, dataloader, intervention_dataloader=None, log_scale=False, absolute=True,
                       apply_greedy=False):
        return evaluate_area_under_curve(
            model=self.model,
            graph=self.graph,
            dataloader=dataloader,
            metrics=self.attribution_metric,
            level=self.level,
            log_scale=log_scale,
            absolute=absolute,
            task=self.task,
            model_name=self.model.cfg.model_name,
            intervention=self.ablation,
            intervention_dataloader=intervention_dataloader,
            optimal_ablation_path=self.optimal_ablation_path,
            apply_greedy=apply_greedy
        )