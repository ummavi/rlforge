#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Get simple information from the sacred mongoDB observer.

TODO: * Remove mongoDB caching to always fetch latest values
      * FileStorage observer as an alternative to MongoDB observer
    
"""

import pymongo
class sacred_db:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["sacred"]

    def get_all_runs(self):
        return self.db.runs.find()

    def run_summaries(self):
        all_runs = []
        for run in self.get_all_runs():
            run_summary = {}
            run_summary["id"] = run["_id"]
            run_summary["name"] = run["experiment"]["name"]
            run_summary["status"] = run["status"]
            run_summary["last_line"] = run["captured_out"][
                run["captured_out"].strip().rfind("\n"):]
            run_summary["config"] = run["config"]
            run_summary["metrics"] = run["info"][
                "metrics"] if "metrics" in run["info"] else []
            all_runs.append(run_summary)
        return all_runs

    def get_experiment_list(self, delete_list=["metrics"]):
        """Get the list of experiments and high level details
        """
        summaries = self.run_summaries()
        all_config_keys = []
        for row in summaries:
            all_config_keys += row["config"].keys()

            #Expand config items (hyperparamters)
            for k, v in row["config"].items():
                row[k] = v
            del row["config"]


            for delete_key in delete_list:
                if delete_key in row:
                    del row[delete_key]

        all_config_keys = set(all_config_keys)
        all_config_keys.remove("seed")
        return summaries, all_config_keys

    def get_metrics(self, run_id):
        """Return all the metrics saved under the run_id
        """
        return list(self.db.metrics.find({"run_id": int(run_id)}))