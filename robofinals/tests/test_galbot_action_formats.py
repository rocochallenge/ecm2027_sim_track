"""Action compatibility checks without launching Isaac Sim."""
import ast
from contextlib import contextmanager
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / 'core/mdp/actions/galbot_base_velocity_action.py'


def make_term(num_envs=1):
    # Exercise the production methods without importing simulator-only modules.
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GalbotBaseVelocityAction')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('adapt_full_action', 'process_actions', 'reset')]
    namespace = {'torch': torch, 'contextmanager': contextmanager, 'Sequence': list}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    term_type = type('TestTerm', (), {n.name: namespace[n.name] for n in methods})
    term = term_type()
    term.device = 'cpu'
    term.num_envs = num_envs
    term._raw_actions = torch.zeros(num_envs, 3)
    term._processed_actions = torch.zeros(num_envs, 3)
    term._smoothing_alpha = .1
    return term


class GalbotActionFormatTests(unittest.TestCase):
    def test_legacy_axis_mapping_and_no_double_smoothing(self):
        term = make_term()
        for wheel, expected in [([.4]*4, [.4, 0, 0]),
                                ([-.4, .4, .4, -.4], [0, .4, 0]),
                                ([-.4, .4, -.4, .4], [0, 0, .4])]:
            action = torch.arange(20, dtype=torch.float32).unsqueeze(0)
            action[:, 16:] = torch.tensor(wheel)
            with term.adapt_full_action(action) as adapted:
                torch.testing.assert_close(adapted[:, :16], action[:, :16])
                term.process_actions(adapted[:, 16:])
                torch.testing.assert_close(term._processed_actions, torch.tensor([expected]))
            self.assertFalse(term._legacy_wheel_input)

    def test_batched_legacy_round_trip(self):
        term = make_term(3)
        velocity = torch.tensor([[.2, -.3, .1], [-.7, .1, -.2], [0., 0., 0.]])
        x, y, z = velocity.unbind(-1)
        wheels = torch.stack((x-y-z, x+y+z, x+y-z, x-y+z), -1)
        with term.adapt_full_action(torch.cat((torch.zeros(3,16), wheels), -1)) as action:
            torch.testing.assert_close(action[:,16:], velocity)

    def test_native_smoothing_reset_and_list_input(self):
        term = make_term()
        for _ in range(2):
            with term.adapt_full_action([0.]*16 + [.4, 0., 0.]) as action:
                term.process_actions(action[:,16:])
            torch.testing.assert_close(term._processed_actions, torch.tensor([[.04,0,0]]))
            term.reset()

    def test_context_cleans_up_after_failure(self):
        term = make_term()
        with self.assertRaises(RuntimeError):
            with term.adapt_full_action(torch.zeros(1,20)):
                raise RuntimeError('simulated step failure')
        self.assertFalse(term._legacy_wheel_input)

    def test_reject_invalid_shapes_and_nonfinite_values(self):
        term = make_term()
        for action in (torch.zeros(1,18), torch.zeros(1,21), torch.zeros(2,19),
                       torch.full((1,19), float('nan'))):
            with self.assertRaises(ValueError):
                with term.adapt_full_action(action):
                    self.fail('invalid action accepted')


if __name__ == '__main__':
    unittest.main()
