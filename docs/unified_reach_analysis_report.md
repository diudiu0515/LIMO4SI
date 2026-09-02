# Unified Reach Analysis Report

This report audits the merged reachability + reach-for logic. It uses shoulder, elbow, wrist, object 3D center, and short temporal wrist trajectory.

- Records: 51
- Answer types: {'unified_reach_analysis': 51}
- Reach states: {'not_reachable_not_reaching': 38, 'reaching_but_not_yet_reachable': 10, 'reachable_hand_close': 2, 'reachable_not_reaching': 1}
- Static reachable: {'False': 48, 'True': 3}
- Reaching motion: {'False': 41, 'True': 10}

## Meaning of states

- `reachable_hand_close`: object is within arm span and a hand is already close.
- `reachable_not_reaching`: object is within arm span, but no clear approach motion.
- `reaching_but_not_yet_reachable`: hand is moving toward the object, but current arm pose cannot yet reach it.
- `not_reachable_not_reaching`: neither static reachability nor reach-for motion is detected.

## Examples

### sfu0083_cam04_3450::02::reachability

- question type: `reachability`
- target: `knife_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The knife_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### sfu0083_cam04_3450::05::current_interaction_object

- question type: `current_interaction_object`
- target: `stainless bowl_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The stainless bowl_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### sfu0083_cam04_3450::08::perspective_reachable_nearest

- question type: `perspective_reachable_nearest`
- target: `stainless bowl_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The stainless bowl_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### sfu0101_cam05_5460::02::reachability

- question type: `reachability`
- target: `cooking oil bottle_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The cooking oil bottle_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### sfu0101_cam05_5460::05::current_interaction_object

- question type: `current_interaction_object`
- target: `white bowl_0`
- state: `reaching_but_not_yet_reachable`
- static reachable: `False`
- reaching motion: `True`

A: The hand is moving toward the white bowl_0, but the object is not yet within the current arm span.

### sfu0101_cam05_5460::08::perspective_reachable_nearest

- question type: `perspective_reachable_nearest`
- target: `white bowl_0`
- state: `reaching_but_not_yet_reachable`
- static reachable: `False`
- reaching motion: `True`

A: The hand is moving toward the white bowl_0, but the object is not yet within the current arm span.

### extra_sfu0083_3510::02::reachability

- question type: `reachability`
- target: `stainless bowl_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The stainless bowl_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0083_3510::05::current_interaction_object

- question type: `current_interaction_object`
- target: `stainless bowl_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The stainless bowl_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0083_3510::08::perspective_reachable_nearest

- question type: `perspective_reachable_nearest`
- target: `stainless bowl_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The stainless bowl_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0083_4200::02::reachability

- question type: `reachability`
- target: `stainless bowl_0`
- state: `reaching_but_not_yet_reachable`
- static reachable: `False`
- reaching motion: `True`

A: The hand is moving toward the stainless bowl_0, but the object is not yet within the current arm span.

### extra_sfu0083_4200::05::current_interaction_object

- question type: `current_interaction_object`
- target: `stainless bowl_0`
- state: `reaching_but_not_yet_reachable`
- static reachable: `False`
- reaching motion: `True`

A: The hand is moving toward the stainless bowl_0, but the object is not yet within the current arm span.

### extra_sfu0083_4200::08::perspective_reachable_nearest

- question type: `perspective_reachable_nearest`
- target: `stainless bowl_0`
- state: `reaching_but_not_yet_reachable`
- static reachable: `False`
- reaching motion: `True`

A: The hand is moving toward the stainless bowl_0, but the object is not yet within the current arm span.

### extra_sfu0101_13920::02::reachability

- question type: `reachability`
- target: `cooking oil bottle_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The cooking oil bottle_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0101_13920::05::current_interaction_object

- question type: `current_interaction_object`
- target: `cooking oil bottle_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The cooking oil bottle_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0101_13920::08::perspective_reachable_nearest

- question type: `perspective_reachable_nearest`
- target: `cooking oil bottle_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The cooking oil bottle_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0101_18510::02::reachability

- question type: `reachability`
- target: `red fork_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The red fork_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0101_18510::05::current_interaction_object

- question type: `current_interaction_object`
- target: `omelet_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The omelet_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### extra_sfu0101_18510::08::perspective_reachable_nearest

- question type: `perspective_reachable_nearest`
- target: `omelet_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The omelet_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### final_sfu0101_13920::02::reachability

- question type: `reachability`
- target: `cooking oil bottle_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The cooking oil bottle_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.

### final_sfu0101_13920::05::current_interaction_object

- question type: `current_interaction_object`
- target: `cooking oil bottle_0`
- state: `not_reachable_not_reaching`
- static reachable: `False`
- reaching motion: `False`

A: The cooking oil bottle_0 is not reachable from the current arm pose, and the hand trajectory does not clearly target it.
