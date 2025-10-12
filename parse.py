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
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        # Augmented start (anchor)
        self._aug = "__START__"


        # --- Viterbi state keyed by (Item, end_col) ---
        # end_col is the column the item currently lives in (i.e., the item's end position)
        self.vit: Dict[Tuple[Item, int], float] = {}
        self.bp_scan: Dict[Tuple[Item, int], Tuple[Tuple[Item, int], str]] = {}
        self.bp_attach: Dict[Tuple[Item, int], Tuple[Tuple[Item, int], Tuple[Item, int]]] = {}
        # best full cost for a completed constituent span (lhs, i, j)
        self.best_complete: Dict[Tuple[str, int, int], float] = {}




        self.cols: List[Agenda]
        self._run_earley()    # run Earley's algorithm to construct self.cols

    def best_root_item(self) -> Optional[Tuple[Item, float]]:
        n = len(self.tokens)
        best_aug = None
        best_cost = None
        for it in self.cols[-1].all():
            if it.rule.lhs == self._aug and it.next_symbol() is None and it.start_position == 0:
                full = self.vit[(it, n)] + it.rule.weight  # aug rule has weight 0
                if best_cost is None or full < best_cost - 1e-12:
                    best_cost = full
                    best_aug = it
        if best_aug is None:
            return None
        # unwrap ROOT child from S' completion
        bp = self.bp_attach.get((best_aug, n))
        if bp is None:
            return None
        _, (root_child, root_end) = bp
        # Return the completed ROOT item with same full cost
        return (root_child, best_cost)





    def _build_tree(self, item: Item, end_col: int):
        """Reconstruct best tree for a completed 'item' ending at end_col."""
        lhs = item.rule.lhs
        kids = []
        key = (item, end_col)
        j = end_col
        while True:
            if key in self.bp_attach:
                (left_item, left_end), (right_item, right_end) = self.bp_attach[key]
                # right child ends at j
                right_tree = self._build_tree(right_item, j)
                kids.append(right_tree)
                # move to left customer, whose end is the start of right child
                j = right_item.start_position
                key = (left_item, j)
            elif key in self.bp_scan:
                (prev_item, prev_end), tok = self.bp_scan[key]
                kids.append((tok,))
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

    def _ensure_item(self, col_index: int, item: Item, cand_cost: float) -> None:
        """Insert or improve an item's score for this column; always ensure it's enqueued in this column."""
        key = (item, col_index)
        prev = self.vit.get(key)
        # Always ensure it's present in this column's agenda
        self.cols[col_index].push(item)
        if prev is None or cand_cost < prev - 1e-12:
            self.vit[key] = cand_cost
            # If we improved an existing entry, requeue so consequences propagate
            if prev is not None:
                self.cols[col_index].requeue(item)


    def _finish_constituent(self, completed: Item, end_col: int) -> float:
        """Record/return full cost for a completed item at its span."""
        i = completed.start_position
        key_span = (completed.rule.lhs, i, end_col)
        full_cost = self.vit[(completed, end_col)] + completed.rule.weight
        best = self.best_complete.get(key_span)
        if best is None or full_cost < best - 1e-12:
            self.best_complete[key_span] = full_cost
        return self.best_complete[key_span]




    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Seed: S' → · ROOT at column 0 (weight 0)
        aug_rule = Rule(self._aug, (self.grammar.start_symbol,), 0.0)
        start_item = Item(aug_rule, 0, 0)
        self.vit[(start_item, 0)] = 0.0
        self.cols[0].push(start_item)

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
            while column:    # while agenda isn't empty
                item = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol();
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)                      

    def _predict(self, nonterminal: str, position: int) -> None:
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule, dot_position=0, start_position=position)
            self._ensure_item(position, new_item, cand_cost=0.0)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1



    def _scan(self, item: Item, position: int) -> None:
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            cand = self.vit[(item, position)]  # terminals don't add weight
            prev = self.vit.get((new_item, position + 1))
            self._ensure_item(position + 1, new_item, cand)
            if prev is None or cand < prev - 1e-12:
                self.bp_scan[(new_item, position + 1)] = ((item, position), self.tokens[position])
            nxt = new_item.next_symbol()
            if nxt is not None and self.grammar.is_nonterminal(nxt):
                self._predict(nxt, position + 1)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1




    def _attach(self, item: Item, position: int) -> None:
        mid = item.start_position
        # ensure child's full cost is finalized for its span (mid..position)
        child_full = self._finish_constituent(item, position)

        for customer in self.cols[mid].all():
            if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                cand = self.vit.get((customer, mid), 0.0) + child_full
                prev = self.vit.get((new_item, position))
                self._ensure_item(position, new_item, cand)
                if prev is None or cand < prev - 1e-12:
                    self.bp_attach[(new_item, position)] = ((customer, mid), (item, position))
                nxt = new_item.next_symbol()
                if nxt is not None and self.grammar.is_nonterminal(nxt):
                    self._predict(nxt, position)
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
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level) 

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                # after: chart = EarleyChart(...)

                if sentence.strip() == "3 * 5":
                    comps = [it for it in chart.cols[-1].all()
                            if it.rule.lhs == args.start_symbol and it.next_symbol() is None and it.start_position == 0]
                    print("#debug completed roots:", len(comps))

                best = chart.best_root_item()
                if best is None:
                    print("NONE")
                else:
                    item, cost = best
                    tree = chart._build_tree(item, end_col=len(sentence.split()))

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
