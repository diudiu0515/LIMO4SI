# Task 1 / Task 3 Expanded QA Quality Report

- Records: 170
- Missing evidence paths: 0
- All have result_json: True
- Confidence: {'high': 51, 'medium': 119}
- Level-2 positive blocker count: 0
- Visibility distribution: {'False': 14, 'True': 20}

## Question type counts
- task1::current_interaction_object: 17
- task1::nearest_referring_object: 17
- task1::quantitative_distance_and_direction: 17
- task1::reachability: 17
- task1::visibility: 17
- task3::level2_perspective_taking: 17
- task3::person_perspective_left_right_front_back: 17
- task3::perspective_reachable_nearest: 17
- task3::perspective_visibility_occlusion: 17
- task3::reference_frame_switching: 17

## Approximations by type
- level2_perspective_taking: 17
- perspective_visibility_occlusion: 17
- visibility: 17

## Missing evidence by type
- reference_frame_switching: 17

## Recommendations
- Keep high-confidence distance/direction and nearest-object QA as core benchmark seed.
- Treat visibility, reachability, interaction, and Level-2 QA as medium-confidence until denser validation is added.
- Actively mine positive Level-2 blocker cases; current expanded pool still has few or none.
- For interaction_object, add temporal hand trajectory/gaze before promoting to high confidence.
