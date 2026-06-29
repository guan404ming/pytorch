# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing


class DynamicParameterTests(torch._dynamo.test_case.TestCase):
    def test_parameter_mark_dynamic_no_recompile(self):
        """mark_dynamic on Parameter should prevent recompilation on shape change."""
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            return x.cos()

        p = torch.nn.Parameter(torch.ones(2, 2))
        for d in range(p.dim()):
            torch._dynamo.mark_dynamic(p, d)
        fn(p)

        # Different shape — should NOT recompile because dims are dynamic
        p2 = torch.nn.Parameter(torch.ones(3, 3))
        for d in range(p2.dim()):
            torch._dynamo.mark_dynamic(p2, d)
        fn(p2)

        self.assertEqual(cnts.frame_count, 1)

    def test_parameter_mark_dynamic_min_only(self):
        """mark_dynamic(x, d, min=0) with max=None should not crash."""
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            return x.cos()

        p = torch.nn.Parameter(torch.ones(2, 2))
        for d in range(p.dim()):
            torch._dynamo.mark_dynamic(p, d, min=0)
        # This should not raise "not simple sympy type <class 'NoneType'>"
        fn(p)
        self.assertEqual(cnts.frame_count, 1)

    def test_parameter_mark_dynamic_min_max_size_one(self):
        """mark_dynamic with min=0, max=65536 on size-(1,1) should not raise CONSTRAINTS_VIOLATED."""
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x):
            return x.sin()

        p = torch.nn.Parameter(torch.ones(1, 1))
        for d in range(p.dim()):
            torch._dynamo.mark_dynamic(p, d, min=0, max=65536)
        # This should not raise ConstraintViolationError
        fn(p)
        self.assertEqual(cnts.frame_count, 1)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
