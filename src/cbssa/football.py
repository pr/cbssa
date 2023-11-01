import enum
import numpy as np


class Down(enum.Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


FIELD_OF_PLAY_LENGTH = 100
FIRST_DOWN_YARDS = 10
LENGTH_OF_END_ZONE = 10
YARDS_BACK_FROM_LINE_OF_SCRIMMAGE = 7
FIELD_GOAL_VALUE = 3
AVERAGE_POINTS_AFTER_KICKOFF = 0.35


def expected_points(
        down: Down,
        yards_to_go: int,
        yards_from_own_goal: int
) -> float:
    if 0 > yards_from_own_goal > FIELD_OF_PLAY_LENGTH:
        raise Exception("Yards from own goal must be between 0 and 100")

    match down:
        case Down.FIRST:
            coefficient_constant = -0.00856283308233047
            coefficient_yards_to_go = -0.0550851328725266
            coefficient_yards_from_own_goal = 0.0660210530697267

            return (
                coefficient_constant
                + yards_to_go * coefficient_yards_to_go
                + coefficient_yards_from_own_goal * yards_from_own_goal
            )

        case Down.SECOND:
            coefficient_constant = -0.303856636504588
            coefficient_yards_to_go = -0.0758606658214685
            coefficient_yards_from_own_goal = 0.0552141997294965
            coefficient_yards_from_own_goal_squared_div_100 = 0.0257402320978258
            coefficient_yards_from_own_goal_cubed_div_1000 = -0.00162392815881755

        case Down.THIRD:
            coefficient_constant = -0.667759263455218
            coefficient_yards_to_go = -0.113922406485868
            coefficient_yards_from_own_goal = 0.0415357137786255
            coefficient_yards_from_own_goal_squared_div_100 = 0.0514312017289563
            coefficient_yards_from_own_goal_cubed_div_1000 = -0.00303269693045725

        case Down.FOURTH:
            coefficient_constant = 3.66629456508042
            coefficient_yards_to_go = -0.00514286252298707
            coefficient_yards_from_own_goal = -0.137033106444474
            coefficient_yards_from_own_goal_squared_div_100 = 0.174435207089797
            coefficient_yards_from_own_goal_cubed_div_1000 = -0.00309304541086363

        case _:
            raise Exception("Invalid Down")

    yards_from_own_goal_squared_div_100 = yards_from_own_goal ** 2 / 100
    yards_from_own_goal_cubed_div_1000 = yards_from_own_goal ** 3 / 1000

    return (
        coefficient_constant
        + yards_to_go * coefficient_yards_to_go
        + coefficient_yards_from_own_goal * yards_from_own_goal
        + coefficient_yards_from_own_goal_squared_div_100 * yards_from_own_goal_squared_div_100
        + coefficient_yards_from_own_goal_cubed_div_1000 * yards_from_own_goal_cubed_div_1000
    )


def fourth_down_conversion_probability(yards_to_go: int) -> float:
    coefficient_yards_to_go = -0.125032858557331
    coefficient_constant = 0.419038612059725
    return 1 - 1 / (1 + np.exp(coefficient_constant + coefficient_yards_to_go * yards_to_go))


def successful_field_goal_probability(distance: int) -> float:
    season = 2016
    coefficient_season = 0.00636803399534301
    coefficient_distance = -0.613136119101955
    coefficient_distance_squared_div_100 = 1.22458102370408
    coefficient_distance_cubed_div_1000 = -0.0953805544332438
    coefficient_constant = 0.000154633519913196

    field_goal_distance_squared_div_100 = distance ** 2 / 100
    field_goal_distance_cubed_div_1000 = distance ** 3 / 1000

    return 1 - 1 / (1 + np.exp(
        coefficient_constant
        + coefficient_season * season
        + coefficient_distance * distance
        + coefficient_distance_squared_div_100 * field_goal_distance_squared_div_100
        + coefficient_distance_cubed_div_1000 * field_goal_distance_cubed_div_1000
    ))


def expected_points_going_for_it_fourth_down(yards_from_own_goal: int, yards_to_go: int) -> float:
    success_yards_from_own_goal = yards_from_own_goal + yards_to_go
    failure_yards_from_own_goal = FIELD_OF_PLAY_LENGTH - yards_from_own_goal

    expected_points_success = expected_points(Down.FIRST, FIRST_DOWN_YARDS, success_yards_from_own_goal)
    expected_points_failure = -expected_points(Down.FIRST, FIRST_DOWN_YARDS, failure_yards_from_own_goal)

    success_probability = fourth_down_conversion_probability(yards_to_go)

    return expected_points_success * success_probability + expected_points_failure * (1 - success_probability)


def expected_points_field_goal(yards_from_own_goal: int) -> float:
    expected_points_missed_field_goal = -expected_points(
        down=Down.FIRST,
        yards_to_go=FIRST_DOWN_YARDS,
        yards_from_own_goal=FIELD_OF_PLAY_LENGTH - yards_from_own_goal + YARDS_BACK_FROM_LINE_OF_SCRIMMAGE
    )
    expected_points_made_field_goal = FIELD_GOAL_VALUE - AVERAGE_POINTS_AFTER_KICKOFF

    success_probability_field_goal = successful_field_goal_probability(
        FIELD_OF_PLAY_LENGTH - yards_from_own_goal + YARDS_BACK_FROM_LINE_OF_SCRIMMAGE + LENGTH_OF_END_ZONE
    )

    return (expected_points_made_field_goal * success_probability_field_goal
            + expected_points_missed_field_goal * (1 - success_probability_field_goal))


def expected_points_punt(yards_from_own_goal: int) -> float:
    coefficient_yards_from_own_goal_squared = -0.000660508110639487
    coefficient_yards_from_own_goal = 0.0894202936275007
    coefficient_constant = -2.92497822631658

    yards_from_own_goal_squared = yards_from_own_goal ** 2

    return (
        coefficient_constant
        + yards_from_own_goal * coefficient_yards_from_own_goal
        + yards_from_own_goal_squared * coefficient_yards_from_own_goal_squared
    )
