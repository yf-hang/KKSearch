"""Core numerical, neural and isolation regression checks."""
import itertools
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from model import CoefficientDistribution, load_policy
from physics import (
    Background, STATES, SECTORS, boson_modes, catalog, finite_volume,
    generate, matrix, shooting_modes, solve_modes,
)
from main import _isolated_training
from utils import load_config

class PhysicsTests(unittest.TestCase):
    def test_flat_scalar_spectrum_and_modes(self):
        bulk = 1.7
        modes = solve_modes(0.0, bulk, 5, 192, 2e-9)
        expected = bulk**2 + (np.pi * np.arange(5))**2
        np.testing.assert_allclose(modes.eigenvalues, expected, rtol=2e-9, atol=1e-9)
        for n in range(5):
            factor = 1 if n == 0 else np.sqrt(2)
            np.testing.assert_allclose(modes.values[n], factor*np.cos(n*np.pi*modes.x), atol=2e-8)

    def test_zero_mode_is_finite_and_normalized(self):
        modes = solve_modes(2.0, 0.0, 4)
        self.assertEqual(modes.eigenvalues[0], 0)
        self.assertEqual(np.max(np.abs(modes.derivatives[0])), 0)
        self.assertLess(modes.gram_error, 1e-8)
        self.assertLess(modes.bc_error, 1e-9)

    def test_finite_volume_converges_towards_separate_bvp(self):
        exact = solve_modes(2.0, 1.2, 5).eigenvalues
        _, coarse, _ = finite_volume(2.0, 1.2, 5, 64)
        _, fine, _ = finite_volume(2.0, 1.2, 5, 256)
        ec = np.max(np.abs(coarse-exact)/np.maximum(exact, 1))
        ef = np.max(np.abs(fine-exact)/np.maximum(exact, 1))
        self.assertLess(ef, ec / 8)

    def test_quadrature_and_tolerance_refinement(self):
        terms = catalog([1, 3, 5, 7], 19)
        tuples = [[0, 1, 2, 3, 4, 5], [2, 0, 3, 1, 5, 4], [0, 2, 3, 1, 4, 5]]
        raw = matrix(3.2, 1.4, tuples, terms, [0], 128, 1e-8)
        fine = matrix(3.2, 1.4, tuples, terms, [0], 256, 2e-12,'independent')
        np.testing.assert_allclose(raw, fine, rtol=2e-7, atol=1e-8)

    def test_invalid_parameters(self):
        for q, b in ((-1, 0), (1, -1), (np.nan, 0), (np.inf, 0)):
            with self.assertRaises(ValueError):
                Background(q, b)

    def test_high_mode_independent_shooting_crosscheck(self):
        q, bulk = 0.71038820502036, 0.7867116245854745
        bvp = solve_modes(q, bulk, 6)
        shot = shooting_modes(q, bulk, 6)
        np.testing.assert_allclose(bvp.eigenvalues, shot.eigenvalues, rtol=1e-8, atol=1e-9)
        self.assertLess(shot.gram_error, 1e-8)


class BosonTests(unittest.TestCase):
    def test_complete_profile_and_primitive_coverage(self):
        terms=catalog()
        self.assertEqual(set(STATES),{'s','u','v','w'})
        self.assertEqual(len(SECTORS),84)
        self.assertEqual(len(terms),396)
        for state in STATES:
            self.assertIn((state,)*6,SECTORS)
        self.assertEqual(len({(t['weight'],tuple(t['derivatives']),tuple(t['factors'])) for t in terms}),396)
        self.assertTrue(all(len(t['derivatives'])+len(t['factors'])==2 for t in terms))

    def test_all_profiles_against_independent_solver(self):
        with threadpool_limits(limits=1):
            for q,b in ((.7,1.2),(2.7,3.1)):
                x,dx,a=boson_modes(q,b,6,384,2e-9)
                _,_,z=boson_modes(q,b,6,384,2e-12,'independent')
                for state in STATES:
                    start=0 if STATES[state]['zero_mode'] else 1
                    f,d,lam=a[state]
                    np.testing.assert_allclose(f[start:],z[state][0][start:],rtol=2e-7,atol=2e-7)
                    np.testing.assert_allclose(d[start:],z[state][1][start:],rtol=2e-7,atol=2e-6)
                    power=3
                    gram=(f[start:]*(dx*(1+q*x)**-power))@f[start:].T
                    np.testing.assert_allclose(gram,np.eye(len(gram)),atol=2e-8)
                self.assertTrue(np.isnan(a['v'][0][0]).all())

    def test_sector_specific_data_and_absent_vector_zero_mode(self):
        cfg=load_config()['data']
        cfg['backgrounds']=dict.fromkeys(cfg['backgrounds'],1)
        cfg['tuples_per_background']=4
        with TemporaryDirectory() as folder, threadpool_limits(limits=1):
            manifest=generate(Path(folder),cfg,catalog())
            with np.load(Path(folder)/'data.npz') as data:
                for split in cfg['backgrounds']:
                    self.assertEqual(data[split+'_matrix'].shape,(84,4,396))
                    self.assertEqual(data[split+'_meta'].shape,(84,4,8))
                    self.assertTrue(np.isfinite(data[split+'_matrix']).all())
                    for i,states in enumerate(SECTORS):
                        for slot,state in enumerate(states):
                            if state=='v': self.assertTrue(np.all(data[split+'_meta'][i,:,slot+2]>0))

    def test_conditioned_policy_roundtrip_and_moments(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        m=CoefficientDistribution(5,[-1,1],contexts=3)
        contexts=torch.tensor([0,1,2,0])
        dist=m.distribution(4,contexts=contexts)
        c,tokens=m.sample_distribution(dist)
        np.testing.assert_array_equal(m.decode(tokens.numpy()),c.numpy())
        probs=dist['probabilities'].gather(-1,tokens[:,2:7,None]).squeeze(-1)
        torch.testing.assert_close(m.log_probability(tokens),dist['control_logprob']+probs.log().sum(-1))
        x=torch.randn(3,8,5,dtype=torch.float64);g=x.transpose(1,2)@x/8
        mean,var,_,_=m.quadratic_moments(dist,g)
        for i in range(4):
            p=dist['probabilities'][i].double();a=torch.tensor(m.alphabet,dtype=torch.float64)
            mu=p@a;variance=p@a.square()-mu.square()
            expected=mu@g[contexts[i]]@mu+(variance*g[contexts[i]].diag()).sum()
            torch.testing.assert_close(mean[i]+var[i],expected)
        same=m.distribution(4,contexts=torch.ones(4,dtype=torch.long))
        batched=m.quadratic_moments(same,g)
        shared=m.quadratic_moments(same,g[1])
        for a,b in zip(batched,shared): torch.testing.assert_close(a,b)

    def test_four_mode_tuples_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'six-mode'):
            matrix(1.,2.,[[1,2,3,4]],catalog())

    def test_premise_projections_hold_on_independently_computed_profiles(self):
        # Reviewer premises are tested separately; never copied into training.
        from validation import projection_rows
        terms=catalog();contexts=[i for i,s in enumerate(SECTORS) if s.count('s') in (2,4)]
        rows=np.array([[1,2,3,4,5,6],[2,4,1,3,6,5],[3,1,4,2,5,6]])
        with threadpool_limits(limits=1):
            x=matrix(1.3,2.1,rows,terms,contexts,384,2e-12,'independent')
        for block,context in zip(x,contexts):
            for label,c in projection_rows(terms,SECTORS[context]):
                scale=np.abs(block)@np.abs(c)
                self.assertLess(np.max(np.abs(block@c)/np.maximum(scale,1e-20)),2e-8,label)

    def test_derivability_is_scale_and_order_invariant_but_not_an_occurrence_decision(self):
        from validation import projection_rows,review
        states=('s','s','u','u','u','u');terms=catalog()
        rows=projection_rows(terms,states)
        c=-7*np.array(rows[0][1])+3*np.array(rows[-1][1])
        assessment=review(terms,states,c.tolist())
        self.assertEqual(assessment['status'],'derived_from_published_premises')
        self.assertFalse(assessment['rejected_as_known'])
        self.assertIsNone(assessment['explicitly_stated_in_references'])
        self.assertIsNone(assessment['accepted_as_new'])
        permutation=np.random.default_rng(11).permutation(len(terms))
        self.assertEqual(review([terms[i] for i in permutation],states,c[permutation].tolist())['status'],'derived_from_published_premises')
        k=next(i for i,t in enumerate(terms) if t['weight']==1)
        c[k]+=1
        result=review(terms,states,c.tolist())
        self.assertEqual(result['status'],'requires_literature_review')
        self.assertFalse(result['accepted_as_new'])


class CoefficientSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conditional_moments_match_enumerated_integer_expectation(self):
        # Synthetic random matrix, no physical relation or coefficient labels.
        torch.manual_seed(73)
        model = CoefficientDistribution(3, [-1, 1])
        dist = model.distribution(1, controls=torch.tensor([[1, 0]]))
        x = torch.randn(7, 3, dtype=torch.float64)
        gram = x.T@x/len(x)
        mean, variance, _, support = model.quadratic_moments(dist, gram)
        expected = torch.zeros((), dtype=torch.float64)
        expected_support = torch.zeros_like(expected)
        for a, b in itertools.product(range(3), repeat=2):
            c = torch.tensor([model.alphabet[a], 1, model.alphabet[b]], dtype=torch.float64)
            probability = dist["probabilities"][0, 0, a]*dist["probabilities"][0, 2, b]
            expected = expected+probability*(c@gram@c)
            expected_support = expected_support+probability*(c != 0).sum()
        torch.testing.assert_close((mean+variance)[0], expected, rtol=2e-7, atol=1e-7)
        torch.testing.assert_close(support[0], expected_support, rtol=2e-7, atol=1e-7)
        (mean+variance+.01*support).mean().backward()
        for parameter in (model.cell.weight_hh, model.coefficient_head.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_sampling_and_log_probability_respect_the_learned_anchor(self):
        torch.manual_seed(17)
        model = CoefficientDistribution(64, range(-8, 9))
        dist = model.distribution(32)
        c, tokens = model.sample_distribution(dist)
        np.testing.assert_array_equal(c.detach().numpy(), model.decode(tokens.detach().numpy()))
        torch.testing.assert_close(c[torch.arange(32), dist["anchor"]], dist["magnitude"])
        self.assertGreater(len(set(dist["anchor"].tolist())), 10)
        probabilities = dist["probabilities"].gather(-1, tokens[:, 2:, None]).squeeze(-1)
        expected = dist["control_logprob"]+probabilities.log().sum(-1)
        torch.testing.assert_close(model.log_probability(tokens), expected)

    def test_worker_learns_from_only_a_random_matrix_and_roundtrips_checkpoint(self):
        config = load_config()
        settings = {**config["search"], "seed": 31, "alphabet": config["coefficients"],
                    "max_terms": 12, "steps": 2, "batch_size": 8, "eligible_contexts": [0,1,2]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            search, audit = _isolated_training(root, np.random.default_rng(17).normal(size=(3, 48, 64)), settings)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["blocked_accesses"], [])
            self.assertEqual(set(audit["project_modules"]),
                             {"train", "model"})
            self.assertEqual(search["worker_inputs"], ["config.json", "matrix.npz:train"])
            self.assertTrue(search["forbidden_modules_absent"])
            self.assertFalse(search["linear_solver_used"])
            self.assertEqual(search["budget"], 48)
            self.assertIsNone(search["first_success_call"])
            self.assertGreater(search["parameter_l2_change"], 0)
            model, checkpoint = load_policy(root/"neural/policy.pt")
            self.assertIsInstance(model, CoefficientDistribution)
            self.assertEqual(checkpoint["completed_steps"], 2)
            with torch.no_grad():
                c, tokens = model.sample(2)
            np.testing.assert_array_equal(c.numpy(), model.decode(tokens.numpy()))


if __name__ == "__main__":
    unittest.main()
