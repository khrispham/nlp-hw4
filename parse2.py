#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple


log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False, maxcost: float = math.inf) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.maxcost = maxcost
        self.profile: CounterType[str] = Counter()

        # Augmented start
        #self._aug = "__START__"


        # cost of each entry in the chart
        self.cost: Dict[Tuple[Item, int], float] = {}

        # backpointers for terminals (from scan) and nonterminals (from attach)
        self.bp_scan: Dict[Tuple[Item, int], Tuple[Tuple[Item, int], str]] = {}
        self.bp_attach: Dict[Tuple[Item, int], Tuple[Tuple[Item, int], Tuple[Item, int]]] = {}

        # best full cost for a completed constituent
        self.best_complete: Dict[Tuple[str, int, int], float] = {}




        self.cols: List[Agenda]
        self._run_earley()    # run Earley's algorithm to construct self.cols

    # check final column for best parse (else return None)
    def best_root_item(self) -> Optional[Tuple[Item, float]]:
        n = len(self.tokens)
        best_item = None
        best_cost = None
        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol and
                item.next_symbol() is None and
                item.start_position == 0):
                full = self.cost[(item, n)] + item.rule.weight
                if best_cost is None or full < best_cost - 1e-12:
                    best_cost = full
                    best_item = item
        return None if best_item is None else (best_item, best_cost)

    # Optimization: left corner filtering
    def _compute_start_set(self, word: str) -> Dict[str, set[str]]:
        """Compute the left-corner start set for word w_j."""
        start_set: Dict[str, set[str]] = {}
        stack = [word]
        visited = set([word])

        # Recursive DFS: for each Y, add Y to start_set(X) for every X in P(Y)
        while stack:
            Y = stack.pop()
            for X in self.grammar.left_parent.get(Y, []):
                start_set.setdefault(X, set()).add(Y)
                if X not in visited:
                    visited.add(X)
                    stack.append(X)
        return start_set



    def _build_tree(self, item: Item, end_col: int):
        """Reconstruct best tree for a completed 'item' ending at end_col."""
        lhs = item.rule.lhs
        kids = []
        key = (item, end_col)
        j = end_col
        #collect all backpointers 
        while True:
            if key in self.bp_attach:
                (left_item, left_end), (right_item, right_end) = self.bp_attach[key]
                right_tree = self._build_tree(right_item, j)
                kids.append(right_tree)
                # move to left customer
                j = right_item.start_position
                key = (left_item, j)
            elif key in self.bp_scan:
                (prev_item, prev_end), token = self.bp_scan[key]
                kids.append((token,))
                j = prev_end
                key = (prev_item, prev_end)
            else:
                break
        kids.reverse()
        return (lhs, *kids)



    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    return True
        return False   # we didn't find any appropriate item

    def _update_item(self, col_index: int, item: Item, cand_cost: float) -> None:
        """Insert or improve an item's score for this column, always ensure it's enqueued in this column."""
        key = (item, col_index)
        prev = self.cost.get(key)
        min_improve = 1e-6

        # --- Iterative deepening / Rehypothesis ---
        # If we previously skipped this item because its cost exceeded the old maxcost,
        # but the current maxcost is now higher, re-enqueue it for reconsideration.
        if cand_cost > self.maxcost:
            # Store it for later reprocessing
            self._deferred = getattr(self, "_deferred", {})
            old = self._deferred.get(key)
            if old is None or cand_cost < old - min_improve:
                self._deferred[key] = cand_cost
            return

        # --- Normal path ---
        existing = any(
            it.rule == item.rule and it.dot_position == item.dot_position
            and it.start_position == item.start_position
            for it in self.cols[col_index]._items[self.cols[col_index]._next:]
        )
        if existing:
            return

        self.cols[col_index].push(item)
        if prev is None or cand_cost < prev - min_improve:
            self.cost[key] = cand_cost
            if prev is not None:
                self.cols[col_index].requeue(item)



    def _finish_constituent(self, completed: Item, end_col: int) -> float:
        """Record/return full cost for a completed item at its span."""
        i = completed.start_position
        key_span = (completed.rule.lhs, i, end_col)
        full_cost = self.cost[(completed, end_col)] + completed.rule.weight
        best = self.best_complete.get(key_span)
        if best is None or full_cost < best - 1e-12:
            self.best_complete[key_span] = full_cost
        return self.best_complete[key_span]




    def _run_earley(self) -> None:
        """Fill in or continue filling in the Earley chart."""
        if not hasattr(self, "cols"):  # first time only
            self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]
            self.cost = {}
            self.bp_scan = {}
            self.bp_attach = {}
            self.best_complete = {}
            self.profile = Counter()
            for rule in self.grammar.expansions(self.grammar.start_symbol):
                start_item = Item(rule, 0, 0)
                self.cost[(start_item, 0)] = 0.0
                self.cols[0].push(start_item)
        
        # Optimization: space time constraint
        start_time = time.time()
        MAX_TIME = 30.0
        MAX_COLUMN_SIZE = 10000

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")

            # Optimization: space time constraint
            #if len(column._items) > MAX_COLUMN_SIZE:
            #    log.warning(f"Column {i} too large ({len(column)} items), aborting.")
            #    return
            #if time.time() - start_time > MAX_TIME:
            #    log.warning(f"Timeout reached at column {i}, aborting parse.")
            #    return
            
            start_set = None
            if i < len(self.tokens):  # no next word at last column
                start_set = self._compute_start_set(self.tokens[i])
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol()
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i, start_set)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)
        # --- Rehypothesize deferred items once maxcost increases ---
        if hasattr(self, "_deferred") and self._deferred:
            for (item, col_index), cand_cost in list(self._deferred.items()):
                if cand_cost <= self.maxcost:
                    self._update_item(col_index, item, cand_cost)
                    del self._deferred[(item, col_index)]                     

    def _predict(self, nonterminal: str, position: int, start_set: Optional[Dict[str, set[str]]] = None) -> None:
        """Predict new items for a nonterminal the given position."""
        
        predicted_here = getattr(self, "_predicted_here", None)
        if predicted_here is None:
            self._predicted_here = predicted_here = set()
        key = (position, nonterminal)
        if key in predicted_here:
            return
        predicted_here.add(key)

        # Optimization: Batch caching
        key = (position, nonterminal)
        if not hasattr(self, "_predict_cache"):
            self._predict_cache: Dict[Tuple[int, str], List[Rule]] = {}
        if key in self._predict_cache:
            rules = self._predict_cache[key]
        else:
            rules = list(self.grammar.expansions(nonterminal))
            self._predict_cache[key] = rules

        # Optimization: Left-corner filtering
        # If start_set is available, only predict if this nonterminal could lead to something
        # consistent with the next word’s left-corner set.
        allowed_Bs = None
        if start_set is not None and nonterminal in start_set:
            allowed_Bs = start_set[nonterminal]

        # Optimization: word specialization
        # delay predictions rather than pruning
        if position < len(self.tokens):
            word = self.tokens[position].lower()
            if hasattr(self.grammar, "can_begin"):
                # Only skip if the grammar definitely
                #  cannot yield this word.
                can = self.grammar.can_begin(nonterminal, word)
                #if can is False:
                #    return

        # predict
        for rule in rules:
            #if allowed_Bs is not None and rule.rhs:
            #    if rule.rhs[0] not in allowed_Bs:
            #        continue
            new_item = Item(rule, dot_position=0, start_position=position)
            self._update_item(position, new_item, cand_cost=0.0)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1


    def _scan(self, item: Item, position: int) -> None:
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            cand = self.cost[(item, position)]
            prev = self.cost.get((new_item, position + 1))
            self._update_item(position + 1, new_item, cand)
            if prev is None or cand < prev - 1e-12:
                self.bp_scan[(new_item, position + 1)] = ((item, position), self.tokens[position])
            next = new_item.next_symbol()
            if next is not None and self.grammar.is_nonterminal(next):
                self._predict(next, position + 1)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1




    def _attach(self, item: Item, position: int) -> None:
        if not hasattr(self, "_completed_spans"):
            self._completed_spans = set()
        span_key = (item.rule.lhs, item.start_position, position)
        if span_key in self._completed_spans:
            return
        self._completed_spans.add(span_key)

        mid = item.start_position
        child_full = self._finish_constituent(item, position)

        for customer in self.cols[mid].all():
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                cand = self.cost.get((customer, mid), 0.0) + child_full
                prev = self.cost.get((new_item, position))
                self._update_item(position, new_item, cand)
                if prev is None or cand < prev - 1e-12:
                    self.bp_attach[(new_item, position)] = ((customer, mid), (item, position))
                next = new_item.next_symbol()
                if next is not None and self.grammar.is_nonterminal(next):
                    self._predict(next, position)
                log.debug(f"\tAttached to get: {new_item} in column {position}")
                self.profile["ATTACH"] += 1




class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    >>> a = Agenda()
    >>> a.push(3)
    >>> a.push(5)
    >>> a.push(3)   # duplicate ignored
    >>> a
    Agenda([]; [3, 5])
    >>> a.pop()
    3
    >>> a
    Agenda([3]; [5])
    >>> a.push(3)   # duplicate ignored
    >>> a.push(7)
    >>> a
    Agenda([3]; [5, 7])
    >>> while a:    # that is, while len(a) != 0
    ...    print(a.pop())
    5
    7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._index: Dict[Item, int] = {}  # stores index of an item if it was ever pushed
        self._next = 0                     # index of first item that has not yet been popped

        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next
    
    def __bool__(self) -> bool:
        return len(self) > 0


    def push(self, item: Item) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item not in self._index:    # O(1) lookup in hash table
            self._items.append(item)
            self._index[item] = len(self._items) - 1
            
    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self)==0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def requeue(self, item: Item) -> None:
        """Force re-processing when an item's score improves (allow a duplicate in the queue)."""
        self._items.append(item)


    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a human-readable string REPResentation of this Agenda."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

        # Optimization: left-corner filtering
        self.prefix: Dict[Tuple[str, str], List[Rule]] = {}      # prefix(A, B) = rules A → B …
        self.left_parent: Dict[str, set[str]] = {}               # left_parent(B) = all A such that A → B …


        for lhs, rules in self._expansions.items():
            for rule in rules:
                if not rule.rhs:  # skip empty right-hand sides
                    continue
                B = rule.rhs[0]
                self.prefix.setdefault((lhs, B), []).append(rule) 
                self.left_parent.setdefault(B, set()).add(lhs)

        # Grammar optimizations
        self.compute_yield_vocab()
        self.compute_can_begin()
        self.collapse_unary()

    def compute_can_begin(self) -> None:
        """Precompute which terminals each nonterminal can yield at its left edge."""
        self.can_begin_table: Dict[str, set[str]] = {}
        for lhs, rules in self._expansions.items():
            starts = set()
            for rule in rules:
                if not rule.rhs:
                    continue
                first = rule.rhs[0]
                if not self.is_nonterminal(first):
                    starts.add(first.lower())
                else:
                    starts.update(self._yield_vocab.get(first, set()))
            self.can_begin_table[lhs] = starts

    def can_begin(self, lhs: str, word: str) -> bool:
        """Return True iff lhs can ultimately yield this terminal (case-insensitive)."""
        table = self.can_begin_table.get(lhs)
        if table is None:
            return True  # unknown lhs → be permissive
        if not table:
            return True  # empty set → unknown, not proven impossible
        return word.lower() in table

    def collapse_unary(self):
        """Pre-collapse unary rules to reduce prediction depth."""
        unary = {lhs: [r.rhs[0] for r in rules if len(r.rhs) == 1 and self.is_nonterminal(r.rhs[0])]
                for lhs, rules in self._expansions.items()}
        for lhs in list(unary.keys()):
            closure = set()
            stack = list(unary[lhs])
            while stack:
                x = stack.pop()
                if x in closure: continue
                closure.add(x)
                stack.extend(unary.get(x, []))
            for z in closure:
                if z != lhs:
                    self._expansions[lhs].append(Rule(lhs, (z,), 0.0))
        # --- Ensure ROOT still has a path to S after collapsing ---
        if "ROOT" in self._expansions and not any("S" in r.rhs for r in self._expansions["ROOT"]):
            self._expansions["ROOT"].append(Rule("ROOT", ("S",), 0.0))




    # Optimization: vocab specialization
    def specialize(self, tokens: List[str]) -> "Grammar":
        """
        Return a specialized copy of this Grammar that omits lexical rules
        unrelated to the current sentence's tokens.
        """

        # --- Create a shallow Grammar clone WITHOUT re-calling __init__ ---
        specialized = Grammar.__new__(Grammar)
        specialized.start_symbol = self.start_symbol
        specialized._expansions = {}

        # Build a lowercase lookup set for tokens
        token_set = {t.lower() for t in tokens}

        # Copy or filter rules
        for lhs, rules in self._expansions.items():
            for rule in rules:
                rhs = rule.rhs
                # Keep lexical rules only if their terminal matches the sentence
                if all(sym not in self._expansions for sym in rhs):
                    if any(sym.lower() in token_set for sym in rhs):
                        specialized._expansions.setdefault(lhs, []).append(rule)
                else:
                    specialized._expansions.setdefault(lhs, []).append(rule)

        # Always preserve start symbol rules
        if self.start_symbol not in specialized._expansions and self.start_symbol in self._expansions:
            specialized._expansions[self.start_symbol] = self._expansions[self.start_symbol]

        # Rebuild optimization indices
        specialized.prefix = {}
        specialized.left_parent = {}
        for lhs, rules in specialized._expansions.items():
            for rule in rules:
                if not rule.rhs:
                    continue
                B = rule.rhs[0]
                specialized.prefix.setdefault((lhs, B), []).append(rule)
                specialized.left_parent.setdefault(B, set()).add(lhs)

        # --- REBUILD cached tables manually ---
        specialized._yield_vocab = {}
        specialized._nullable = set()
        specialized.compute_yield_vocab()
        specialized.compute_can_begin()
        specialized.collapse_unary()

        return specialized


    # Optimization: vocab specialization
    def compute_yield_vocab(self) -> None:
        """Precompute the set of terminal symbols that each nonterminal can eventually produce."""
        self._yield_vocab: Dict[str, set[str]] = {lhs: set() for lhs in self._expansions}

        # Initialize with direct terminals
        changed = True
        while changed:
            changed = False
            for lhs, rules in self._expansions.items():
                before = len(self._yield_vocab[lhs])
                for rule in rules:
                    for sym in rule.rhs:
                        if sym not in self._expansions:  # terminal
                            self._yield_vocab[lhs].add(sym.lower())
                        else:
                            self._yield_vocab[lhs].update(self._yield_vocab[sym])
                if len(self._yield_vocab[lhs]) > before:
                    changed = True

        self._nullable: set[str] = set()

        # Compute nullable symbols
        changed = True
        while changed:
            changed = False
            for lhs, rules in self._expansions.items():
                for rule in rules:
                    if all(sym in self._nullable or sym not in self._expansions for sym in rule.rhs):
                        if lhs not in self._nullable:
                            self._nullable.add(lhs)
                            changed = True



    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us declare that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.

    >>> r = Rule('S',('NP','VP'),3.14)
    >>> r
    S → NP VP
    >>> r.weight
    3.14
    >>> r.weight = 2.718
    Traceback (most recent call last):
    dataclasses.FrozenInstanceError: cannot assign to field 'weight'
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        # Note: You might want to modify this to include the weight.
        return f"{self.lhs} → {' '.join(self.rhs)}"

    
# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        # Note: If you revise this class to change what an Item stores, you'll probably want to change this method too.
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides


def main():
    
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f:
            sentence = sentence.strip()
            if not sentence:
                continue

            log.debug("=" * 70)
            log.debug(f"Parsing sentence: {sentence}")

            tokens = sentence.split()
            """
            maxcost = 5.0  # initial cutoff
            best = None
            while True:
                log.info(f"Trying maxcost={maxcost} for: {sentence}")
                # Optimization: vocab specialization
                specialized_grammar = grammar.specialize(tokens)
                chart = EarleyChart(tokens, specialized_grammar, progress=args.progress, maxcost=maxcost)

                best = chart.best_root_item()
                if best is not None:
                    break
                if maxcost > 320:
                    break
                maxcost *= 2  # widen maxcost gradually

            if best is None:
                print("NONE")
            else:
                item, cost = best
                tree = chart._build_tree(item, end_col=len(tokens))
                def show(t):
                    if len(t) == 1:
                        return t[0]
                    return "(" + " ".join([t[0]] + [show(c) for c in t[1:]]) + ")"
                print(show(tree))
                print(cost)
            """

            maxcost = 5.0
            best = None
            chart = None
            specialized_grammar = grammar.specialize(tokens)

            while True:
                log.info(f"Trying maxcost={maxcost} for: {sentence}")

                # Reuse previous chart if possible
                if chart is None:
                    chart = EarleyChart(tokens, specialized_grammar, progress=args.progress, maxcost=maxcost)
                else:
                    # Re-run using the same grammar and existing chart state
                    chart.maxcost = maxcost
                    chart._run_earley()  # resumes parsing with higher cost allowance

                best = chart.best_root_item()
                if best is not None:
                    break
                if math.isinf(maxcost): #or maxcost >= 640:
                    break

                maxcost *= 2

            if best is None:
                print("NONE")
            else:
                item, cost = best
                tree = chart._build_tree(item, end_col=len(tokens))
                def show(t):
                    if len(t) == 1:
                        return t[0]
                    return "(" + " ".join([t[0]] + [show(c) for c in t[1:]]) + ")"
                print(show(tree))
                print(cost)



    


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
