"""
Fuzzy ART Model and Choice Function Components

This module implements:
1. The Fuzzy ART clustering algorithm with customizable choice functions
2. All primitive operations used in the grammar (Appendix A.6)
3. Expression parser and evaluator for generated choice functions
"""

from typing import Dict, Callable, Any, List, Optional, Tuple, Union
import numpy as np
from functools import reduce
import re


class FuzzyART:
    """
    Fuzzy ART (Adaptive Resonance Theory) clustering algorithm.

    Fuzzy ART is an unsupervised learning algorithm that incrementally
    creates clusters (categories) based on pattern similarity.

    The choice function determines which category best matches an input pattern.
    This implementation allows custom choice functions to be used.
    """

    def __init__(self,
                 alpha: float = 0.01,
                 beta: float = 1.0,
                 rho: float = 0.5,
                 choice_function: Optional[Callable] = None):
        """
        Initialize Fuzzy ART.

        Args:
            alpha: Choice parameter (small positive value to break ties)
            beta: Learning rate (0 to 1, 1 = fast learning)
            rho: Vigilance parameter (0 to 1, higher = more categories)
            choice_function: Custom choice function, defaults to standard Fuzzy ART
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        # Use custom choice function or default
        if choice_function is not None:
            self.choice_function = choice_function
        else:
            self.choice_function = self._default_choice_function

        # Weights (category prototypes)
        self.weights: List[np.ndarray] = []

        # Training statistics
        self.num_categories = 0
        self.iterations = 0

    def _default_choice_function(self, f: np.ndarray, w: np.ndarray,
                                 context: Dict[str, Any]) -> float:
        """
        Default Fuzzy ART choice function.

        T_j = |I ∧ w_j| / (α + |w_j|)

        where ∧ is the fuzzy AND (element-wise minimum).

        Args:
            f: Input pattern (complement coded)
            w: Category weight vector
            context: Additional context (alpha, beta, rho, etc.)

        Returns:
            Choice value for this category
        """
        alpha = context.get('alpha', self.alpha)

        # Fuzzy AND (element-wise minimum)
        fuzzy_and = np.minimum(f, w)

        # L1 norm
        numerator = np.sum(np.abs(fuzzy_and))
        denominator = alpha + np.sum(np.abs(w))

        return numerator / denominator if denominator > 0 else 0.0

    def _match_function(self, f: np.ndarray, w: np.ndarray) -> float:
        """
        Compute match value for vigilance test.

        M_j = |I ∧ w_j| / |I|

        Args:
            f: Input pattern
            w: Category weight vector

        Returns:
            Match value
        """
        fuzzy_and = np.minimum(f, w)
        numerator = np.sum(np.abs(fuzzy_and))
        denominator = np.sum(np.abs(f))

        return numerator / denominator if denominator > 0 else 0.0

    def _complement_code(self, x: np.ndarray) -> np.ndarray:
        """
        Apply complement coding to input.

        Complement coding: I = [x, 1-x]
        This ensures that the input norm is constant.

        Args:
            x: Input pattern (values in [0, 1])

        Returns:
            Complement coded pattern
        """
        # Normalize to [0, 1] if needed
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        return np.concatenate([x_norm, 1 - x_norm])

    def fit(self, X: np.ndarray, max_iterations: int = 100) -> 'FuzzyART':
        """
        Fit Fuzzy ART to data.

        Args:
            X: Input data (n_samples, n_features)
            max_iterations: Maximum number of passes through data

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # Initialize weights list
        self.weights = []
        self.num_categories = 0

        # Context for choice function
        context = {
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'iter': 0,
            'num_w': 0
        }

        for iteration in range(max_iterations):
            context['iter'] = iteration
            categories_changed = False

            for i in range(n_samples):
                # Complement code the input
                f = self._complement_code(X[i])

                # If no categories exist, create first one
                if len(self.weights) == 0:
                    self.weights.append(f.copy())
                    self.num_categories = 1
                    categories_changed = True
                    continue

                context['num_w'] = len(self.weights)

                # Compute choice values for all categories
                choice_values = []
                for j, w in enumerate(self.weights):
                    context['cluster'] = j
                    context['text'] = f  # Input pattern
                    cv = self.choice_function(f, w, context)
                    choice_values.append(cv)

                # Sort categories by choice value (descending)
                sorted_indices = np.argsort(choice_values)[::-1]

                # Find resonant category
                resonant_found = False
                for j in sorted_indices:
                    w = self.weights[j]

                    # Vigilance test
                    match_value = self._match_function(f, w)

                    if match_value >= self.rho:
                        # Resonance - update weights
                        self.weights[j] = self.beta * np.minimum(f, w) + (1 - self.beta) * w
                        resonant_found = True
                        break

                # If no resonant category, create new one
                if not resonant_found:
                    self.weights.append(f.copy())
                    self.num_categories = len(self.weights)
                    categories_changed = True

            self.iterations = iteration + 1

            # Early stopping if no changes
            if not categories_changed:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for data.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Cluster labels
        """
        labels = []

        context = {
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'iter': self.iterations,
            'num_w': len(self.weights)
        }

        for i in range(len(X)):
            f = self._complement_code(X[i])

            if len(self.weights) == 0:
                labels.append(-1)
                continue

            # Compute choice values
            choice_values = []
            for j, w in enumerate(self.weights):
                context['cluster'] = j
                context['text'] = f
                cv = self.choice_function(f, w, context)
                choice_values.append(cv)

            # Assign to category with highest choice value
            labels.append(np.argmax(choice_values))

        return np.array(labels)

    def fit_predict(self, X: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X, max_iterations)
        return self.predict(X)


# =============================================================================
# Primitive Operations for Grammar (Appendix A.6)
# =============================================================================

class ChoiceFunctionPrimitives:
    """
    Collection of primitive operations used in the choice function grammar.

    Categories:
    - arth: Arithmetic operations
    - la: Linear algebra operations
    - trig: Trigonometric functions
    - stat: Statistical functions
    - bool: Boolean/fuzzy operations
    """

    # ========== Arithmetic Operations (arth) ==========

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition."""
        return np.add(a, b)

    @staticmethod
    def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise subtraction."""
        return np.subtract(a, b)

    @staticmethod
    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication."""
        return np.multiply(a, b)

    @staticmethod
    def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise division (safe)."""
        return np.divide(a, b + 1e-10)

    @staticmethod
    def min_op(a: np.ndarray) -> float:
        """Minimum value."""
        return np.min(a)

    @staticmethod
    def max_op(a: np.ndarray) -> float:
        """Maximum value."""
        return np.max(a)

    @staticmethod
    def exp(a: np.ndarray) -> np.ndarray:
        """Exponential (clipped to prevent overflow)."""
        return np.exp(np.clip(a, -100, 100))

    # ========== Linear Algebra Operations (la) ==========

    @staticmethod
    def inner(a: np.ndarray, b: np.ndarray) -> float:
        """Inner product."""
        return np.inner(a.flatten(), b.flatten())

    @staticmethod
    def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Outer product."""
        return np.outer(a.flatten(), b.flatten())

    @staticmethod
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""
        try:
            return np.matmul(a, b)
        except ValueError:
            return a * b  # Fallback to element-wise

    @staticmethod
    def dot(a: np.ndarray, b: np.ndarray) -> float:
        """Dot product."""
        return np.dot(a.flatten(), b.flatten())

    @staticmethod
    def l1norm(a: np.ndarray) -> float:
        """L1 norm (sum of absolute values)."""
        return np.sum(np.abs(a))

    @staticmethod
    def l2norm(a: np.ndarray) -> float:
        """L2 norm (Euclidean norm)."""
        return np.linalg.norm(a)

    # ========== Trigonometric Functions (trig) ==========

    @staticmethod
    def sin(a: np.ndarray) -> np.ndarray:
        """Sine."""
        return np.sin(a)

    @staticmethod
    def cos(a: np.ndarray) -> np.ndarray:
        """Cosine."""
        return np.cos(a)

    @staticmethod
    def tan(a: np.ndarray) -> np.ndarray:
        """Tangent (clipped to prevent infinity)."""
        return np.clip(np.tan(a), -1e10, 1e10)

    @staticmethod
    def log(a: np.ndarray) -> np.ndarray:
        """Natural logarithm (safe)."""
        return np.log(np.abs(a) + 1e-10)

    @staticmethod
    def asin(a: np.ndarray) -> np.ndarray:
        """Arcsine (clipped to valid domain)."""
        return np.arcsin(np.clip(a, -1, 1))

    @staticmethod
    def acos(a: np.ndarray) -> np.ndarray:
        """Arccosine (clipped to valid domain)."""
        return np.arccos(np.clip(a, -1, 1))

    @staticmethod
    def atan(a: np.ndarray) -> np.ndarray:
        """Arctangent."""
        return np.arctan(a)

    # ========== Statistical Functions (stat) ==========

    @staticmethod
    def stdev(a: np.ndarray) -> float:
        """Standard deviation."""
        return np.std(a)

    @staticmethod
    def variance(a: np.ndarray) -> float:
        """Variance."""
        return np.var(a)

    @staticmethod
    def average(a: np.ndarray) -> float:
        """Average (mean)."""
        return np.mean(a)

    @staticmethod
    def median(a: np.ndarray) -> float:
        """Median."""
        return np.median(a)

    # ========== Boolean/Fuzzy Operations (bool) ==========

    @staticmethod
    def logical_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Logical AND (boolean)."""
        return np.logical_and(a > 0.5, b > 0.5).astype(float)

    @staticmethod
    def logical_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Logical OR (boolean)."""
        return np.logical_or(a > 0.5, b > 0.5).astype(float)

    @staticmethod
    def logical_not(a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
        """Logical NOT (boolean)."""
        return np.logical_not(a > 0.5).astype(float)

    @staticmethod
    def logical_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Logical XOR (boolean)."""
        return np.logical_xor(a > 0.5, b > 0.5).astype(float)

    @staticmethod
    def fuzzy_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fuzzy AND (element-wise minimum)."""
        return np.minimum(a, b)

    @staticmethod
    def fuzzy_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fuzzy OR (element-wise maximum)."""
        return np.maximum(a, b)


# Operation registry for expression evaluation
OPERATION_REGISTRY: Dict[str, Callable] = {
    # Arithmetic
    'add': ChoiceFunctionPrimitives.add,
    'sub': ChoiceFunctionPrimitives.sub,
    'mul': ChoiceFunctionPrimitives.mul,
    'div': ChoiceFunctionPrimitives.div,
    'min': ChoiceFunctionPrimitives.min_op,
    'max': ChoiceFunctionPrimitives.max_op,
    'exp': ChoiceFunctionPrimitives.exp,

    # Linear algebra
    'inner': ChoiceFunctionPrimitives.inner,
    'outer': ChoiceFunctionPrimitives.outer,
    'matmul': ChoiceFunctionPrimitives.matmul,
    'dot': ChoiceFunctionPrimitives.dot,
    'l1norm': ChoiceFunctionPrimitives.l1norm,
    'l2norm': ChoiceFunctionPrimitives.l2norm,

    # Trigonometric
    'sin': ChoiceFunctionPrimitives.sin,
    'cos': ChoiceFunctionPrimitives.cos,
    'tan': ChoiceFunctionPrimitives.tan,
    'log': ChoiceFunctionPrimitives.log,
    'asin': ChoiceFunctionPrimitives.asin,
    'acos': ChoiceFunctionPrimitives.acos,
    'atan': ChoiceFunctionPrimitives.atan,

    # Statistical
    'stdev': ChoiceFunctionPrimitives.stdev,
    'variance': ChoiceFunctionPrimitives.variance,
    'average': ChoiceFunctionPrimitives.average,
    'median': ChoiceFunctionPrimitives.median,

    # Boolean/Fuzzy
    'logical_and': ChoiceFunctionPrimitives.logical_and,
    'logical_or': ChoiceFunctionPrimitives.logical_or,
    'logical_not': ChoiceFunctionPrimitives.logical_not,
    'logical_xor': ChoiceFunctionPrimitives.logical_xor,
    'fuzzy_and': ChoiceFunctionPrimitives.fuzzy_and,
    'fuzzy_or': ChoiceFunctionPrimitives.fuzzy_or
}


class ExpressionEvaluator:
    """
    Evaluates choice function expressions generated by GE.

    Parses and executes expressions like:
    - "div(l1norm(fuzzy_and(f, cluster)), add(alpha, l1norm(cluster)))"
    - "add(inner(f, cluster), mul(alpha, l2norm(f)))"
    """

    def __init__(self):
        """Initialize the expression evaluator."""
        self.operations = OPERATION_REGISTRY

    def parse_and_evaluate(self, expression: str,
                          context: Dict[str, Any]) -> Union[float, np.ndarray]:
        """
        Parse and evaluate an expression.

        Args:
            expression: String expression to evaluate
            context: Dictionary containing variable values (f, cluster, alpha, etc.)

        Returns:
            Evaluated result
        """
        try:
            result = self._evaluate(expression, context)
            # Return scalar if possible
            if isinstance(result, np.ndarray):
                if result.size == 1:
                    return float(result.flat[0])
                return float(np.mean(result))
            return float(result)
        except Exception as e:
            # Return default value on error
            return 0.0

    def _evaluate(self, expr: str, context: Dict[str, Any]) -> Any:
        """Recursively evaluate an expression."""
        expr = expr.strip()

        # Check if it's a terminal (variable or constant)
        if expr in context:
            return context[expr]

        # Check if it's a number
        try:
            return float(expr)
        except ValueError:
            pass

        # Parse function call: func_name(args)
        match = re.match(r'(\w+)\((.*)\)$', expr, re.DOTALL)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = self._parse_arguments(args_str)

            # Evaluate arguments recursively
            eval_args = [self._evaluate(arg, context) for arg in args]

            # Call operation
            if func_name in self.operations:
                return self.operations[func_name](*eval_args)

        # Return default if can't parse
        return np.array([0.0])

    def _parse_arguments(self, args_str: str) -> List[str]:
        """Parse comma-separated arguments, respecting nested parentheses."""
        args = []
        current = []
        depth = 0

        for char in args_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            args.append(''.join(current).strip())

        return args


def create_choice_function(expression: str) -> Callable:
    """
    Create a callable choice function from an expression string.

    Args:
        expression: Generated expression string

    Returns:
        Callable that takes (f, w, context) and returns choice value
    """
    evaluator = ExpressionEvaluator()

    def choice_function(f: np.ndarray, w: np.ndarray,
                       context: Dict[str, Any]) -> float:
        """Generated choice function."""
        # Build evaluation context
        eval_context = {
            'f': f,
            'text': f,  # Alias for input pattern
            'cluster': w,  # Category weight
            'alpha': context.get('alpha', 0.01),
            'beta': context.get('beta', 1.0),
            'rho': context.get('rho', 0.5),
            'iter': context.get('iter', 0),
            'num_w': context.get('num_w', 1),
            'pi': np.pi
        }

        return evaluator.parse_and_evaluate(expression, eval_context)

    return choice_function
