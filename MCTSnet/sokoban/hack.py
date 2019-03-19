import gym_sokoban
import numpy as np
import marshal

solution = []


def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300, actions=[]):
    """
    Searches through all possible states of the room.
    This is a recursive function, which stops if the tll is reduced to 0 or
    over 1.000.000 states have been explored.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param box_swaps:
    :param last_pull:
    :param ttl:
    :return:
    """
    global solution

    ttl -= 1
    if ttl <= 0 or len(gym_sokoban.envs.room_utils.explored_states) >= 300000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if not (state_tohash in gym_sokoban.envs.room_utils.explored_states):

        # Add current state and its score to explored states
        room_score = box_swaps * gym_sokoban.envs.room_utils.box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != gym_sokoban.envs.room_utils.num_boxes:
            room_score = 0

        if room_score > gym_sokoban.envs.room_utils.best_room_score:
            gym_sokoban.envs.room_utils.best_room = room_state
            gym_sokoban.envs.room_utils.best_room_score = room_score
            gym_sokoban.envs.room_utils.best_box_mapping = box_mapping
            solution = actions.copy()

        gym_sokoban.envs.room_utils.explored_states.add(state_tohash)

        for action in gym_sokoban.envs.room_utils.ACTION_LOOKUP.keys():
            # The state and box mapping  need to be copied to ensure
            # every action start from a similar state.
            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            actions_next = actions.copy()
            actions_next.append(action)

            room_state_next, box_mapping_next, last_pull_next = gym_sokoban.envs.room_utils.reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            depth_first_search(room_state_next, room_structure,
                               box_mapping_next, box_swaps_next, last_pull, ttl, actions_next)


setattr(gym_sokoban.envs.room_utils, depth_first_search.__name__, depth_first_search)
