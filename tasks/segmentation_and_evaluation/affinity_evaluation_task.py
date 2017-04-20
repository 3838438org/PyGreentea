import os

class AlreadyEvaluatedException(Exception):
    pass


class AffinityEvaluationTask(object):
    def __init__(self, affinity_evaluator, data_source, thresholds):
        self.affinity_evaluator = affinity_evaluator
        self.data_source = data_source
        self.thresholds = thresholds

    def run(self):
        filename = "evaluation__{}__{}.csv".format(self.data_source.data_name, self.affinity_evaluator.name)
        output_path = self.data_source.model.outputs_path
        path = os.path.join(output_path, filename)
        if os.path.exists(path):
            raise AlreadyEvaluatedException 
        evaluation_result = self.affinity_evaluator.segment_and_evaluate(self.data_source.affinity, self.data_source.truth, self.thresholds)
        df = evaluation_result.to_dataframe()
        df["model"] = self.data_source.model.name
        df["iteration"] = self.data_source.model.iteration
        df["data_name"] = self.data_source.data_name
        df["truth"] = self.data_source.data_name
        # from datetime import datetime
        # timestr = datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3]
        # output_path = "/groups/turaga/home/grisaitisw/src/PyGreentea/tasks/segmentation_and_evaluation/results"
        df.to_csv(path)
        print("saved to", filename)
        print(evaluation_result.duration, "seconds to segment and evaluate", self.data_source.model.description, "with", evaluation_result.method_name)
