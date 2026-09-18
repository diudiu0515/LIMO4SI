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


    def test_orders_task4_before_task5_without_reordering_within_tasks(self):
        data = {
            "tasks": [{"id": TASK4_ID}, {"id": TASK5_ID}],
            "groups": [
                group("task4-first", TASK4_ID),
                group("task5-first", TASK5_ID),
                group("task4-second", TASK4_ID),
                group("task5-second", TASK5_ID),
            ],
        }

        public = filter_public_data(data)

        self.assertEqual(
            [row["name"] for row in public["groups"]],
            ["task4-first", "task4-second", "task5-first", "task5-second"],
        )


if __name__ == "__main__":
    unittest.main()
