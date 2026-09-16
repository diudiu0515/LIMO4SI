import unittest

from project_task4_task5_release import filter_public_data
from limo4si.scale_quality import TASK4_ID, TASK5_ID


def group(name, task_id):
    return {"name": name, "qa": [{"task_id": task_id}]}


class Task4Task5ProjectionTests(unittest.TestCase):
    def test_removes_task1_and_task3(self):
        data = {
            "tasks": [
                {"id": "task1_dynamic_human_referenced_relations"},
                {"id": "task3_human_scene_topological_reasoning"},
                {"id": TASK4_ID},
                {"id": TASK5_ID},
            ],
            "groups": [
                group("one", "task1_dynamic_human_referenced_relations"),
                group("three", "task3_human_scene_topological_reasoning"),
            ],
        }
        public = filter_public_data(data)
        self.assertEqual(public["groups"], [])
        self.assertEqual({row["id"] for row in public["tasks"]}, {TASK4_ID, TASK5_ID})


if __name__ == "__main__":
    unittest.main()
