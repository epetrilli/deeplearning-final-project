import logging
import numpy as np
from collections import namedtuple

TRACK_NAME = 'icy_soccer_field'
MAX_FRAMES = 1000

RunnerInfo = namedtuple('RunnerInfo', ['agent_type', 'error', 'total_act_time'])


class AIRunner:
    agent_type = 'state'
    is_ai = True

    def new_match(self, team: int, num_players: int) -> list:
        pass

    def act(self, player_state, opponent_state, world_state):
        return []

    def info(self):
        return RunnerInfo('state', None, 0)


class TeamRunner:
    agent_type = 'state'
    _error = None
    _total_act_time = 0

    def __init__(self, team_or_dir):
        from pathlib import Path
        try:
            from grader import grader
        except ImportError:
            try:
                from . import grader
            except ImportError:
                import grader

        self._error = None
        self._team = None
        try:
            if isinstance(team_or_dir, (str, Path)):
                assignment = grader.load_assignment(team_or_dir)
                if assignment is None:
                    self._error = 'Failed to load submission.'
                else:
                    self._team = assignment.Team()
            else:
                self._team = team_or_dir
        except Exception as e:
            self._error = 'Failed to load submission: {}'.format(str(e))
        if hasattr(self, '_team') and self._team is not None:
            self.agent_type = self._team.agent_type

    def new_match(self, team: int, num_players: int) -> list:
        self._total_act_time = 0
        self._error = None
        try:
            r = self._team.new_match(team, num_players)
            if isinstance(r, str) or isinstance(r, list) or r is None:
                return r
            self._error = 'new_match needs to return kart names as a str, list, or None. Got {!r}!'.format(r)
        except Exception as e:
            self._error = 'Failed to start new_match: {}'.format(str(e))
        return []

    def act(self, player_state, *args, **kwargs):
        from time import time
        t0 = time()
        try:
            r = self._team.act(player_state, *args, **kwargs)
        except Exception as e:
            self._error = 'Failed to act: {}'.format(str(e))
        else:
            self._total_act_time += time()-t0
            return r
        return []

    def info(self):
        return RunnerInfo(self.agent_type, self._error, self._total_act_time)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from os import environ
    from . import remote, utils

    parser = ArgumentParser(description="Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('-r', '--record_video', help="Do you want to record a video?")
    parser.add_argument('-s', '--record_state', help="Do you want to pickle the state?")
    parser.add_argument('-f', '--num_frames', default=1200, type=int, help="How many steps should we play for?")
    parser.add_argument('-p', '--num_players', default=2, type=int, help="Number of players per team")
    parser.add_argument('-m', '--max_score', default=3, type=int, help="How many goal should we play to?")
    parser.add_argument('-j', '--parallel', type=int, help="How many parallel process to use?")
    parser.add_argument('--ball_location', default=[0, 0], type=float, nargs=2, help="Initial xy location of ball")
    parser.add_argument('--ball_velocity', default=[0, 0], type=float, nargs=2, help="Initial xy velocity of ball")
    parser.add_argument('team1', help="Python module name or `AI` for AI players.")
    parser.add_argument('team2', help="Python module name or `AI` for AI players.")
    args = parser.parse_args()

    logging.basicConfig(level=environ.get('LOGLEVEL', 'WARNING').upper())

    if args.parallel is None or remote.ray is None:
        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else TeamRunner(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else TeamRunner(args.team2)

        # What should we record?
        recorder = None
        if args.record_video:
            recorder = recorder & utils.VideoRecorder(args.record_video)

        if args.record_state:
            recorder = recorder & utils.StateRecorder(args.record_state)

        # Start the match
        match = Match(use_graphics=team1.agent_type == 'image' or team2.agent_type == 'image') # CHANGE TO GYMNASIUM
        try:
            result = match.run(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                               initial_ball_location=args.ball_location, initial_ball_velocity=args.ball_velocity,
                               record_fn=recorder)
        except MatchException as e:
            print('Match failed', e.score)
            print(' T1:', e.msg1)
            print(' T2:', e.msg2)

        print('Match results', result)

    else:
        # Fire up ray
        remote.init(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()), configure_logging=True,
                    log_to_driver=True, include_dashboard=False)

        # Create the teams
        team1 = AIRunner() if args.team1 == 'AI' else remote.RayTeamRunner.remote(args.team1)
        team2 = AIRunner() if args.team2 == 'AI' else remote.RayTeamRunner.remote(args.team2)
        team1_type, *_ = team1.info() if args.team1 == 'AI' else remote.get(team1.info.remote())
        team2_type, *_ = team2.info() if args.team2 == 'AI' else remote.get(team2.info.remote())

        # What should we record?
        assert args.record_state is None or args.record_video is None, "Cannot record both video and state in parallel mode"

        # Start the match
        results = []
        for i in range(args.parallel):
            recorder = None
            if args.record_video:
                ext = Path(args.record_video).suffix
                recorder = remote.RayVideoRecorder.remote(args.record_video.replace(ext, f'.{i}{ext}'))
            elif args.record_state:
                ext = Path(args.record_state).suffix
                recorder = remote.RayStateRecorder.remote(args.record_state.replace(ext, f'.{i}{ext}'))

            match = remote.RayMatch.remote(logging_level=getattr(logging, environ.get('LOGLEVEL', 'WARNING').upper()),
                                           use_graphics=team1_type == 'image' or team2_type == 'image')
            result = match.run.remote(team1, team2, args.num_players, args.num_frames, max_score=args.max_score,
                                      initial_ball_location=args.ball_location,
                                      initial_ball_velocity=args.ball_velocity,
                                      record_fn=recorder)
            results.append(result)

        for result in results:
            try:
                result = remote.get(result)
            except (remote.RayMatchException, MatchException) as e:
                print('Match failed', e.score)
                print(' T1:', e.msg1)
                print(' T2:', e.msg2)

            print('Match results', result)

class MatchException(Exception):
    def __init__(self, score, msg1, msg2):
        self.score, self.msg1, self.msg2 = score, msg1, msg2
