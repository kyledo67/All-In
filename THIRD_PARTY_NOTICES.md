# Third-party attribution

Dynamax, https://github.com/probml/dynamax, pinned commit
`ffc626bf89bf07c7ae7064922f6a373d013c4b17`, is reused under its MIT license.
The complete license remains at `research/vendor/dynamax/LICENSE` and its citation
metadata at `research/vendor/dynamax/CITATION.cff`. Authors include Scott W.
Linderman, Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li,
Gerardo Duran-Martin and Kevin Murphy; refer to the preserved upstream citation.
The All-In emission extension is identified in `artifacts/dynamax.patch`.
The original categorical file retains its upstream identity; the license applies
to that preserved copy as well.

REA documentation, https://github.com/morluto/rea, pinned commit
`015a02d57b567bc9a1ed792d1e4b3ec083fa0ae3`, is preserved for investigation provenance.
Its MIT license remains in `research/rea_reference/LICENSE`. No REA runtime or
native-analysis engine is installed or needed for this model.

Organizer starter code, loaders, scorer and submission writer are copied without
modification from the released main-participants-60-40 kit. The provided kit does
not contain a separate LICENSE file; no additional license is asserted here.
Its original notices, rules and reproduction contract are retained in `organizer/`.

Training libraries are installed separately using `requirements-training.lock`;
their distributions carry their own notices and licenses. The final shared-model
inference uses Python's standard library and the supplied organizer reader/writer.

## Built with PriorLabs-TabPFN

TabPFN, https://github.com/priorlabs/tabpfn, pinned to `tabpfn==2.2.1`, is reused for
the optional TabPFN blend component. **It is not Apache-2.0.** It is distributed under
the **Prior Labs License Version 1.1 (May 2025)**, an Apache-2.0 derivative whose added
Paragraph 10 requires that anyone distributing or making available the work, its model
weights, or a derivative of either must (A) ship a copy of that license and (B)
prominently display "Built with PriorLabs-TabPFN". The complete license is preserved at
`research/tabpfn/LICENSE`, and this heading satisfies (B) for this repository.

Authors include Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar,
Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter, Eddie Bergman,
Leo Grinsztajn, Felix Jabloski, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund,
Brendan Roof, Dominik Safaric and Benjamin Jaeger.

Paragraph 10 also exempts "internal benchmarking and testing without external
communication". Measuring the component against our own blend on public validation is
that exempt case. Shipping the 28 MB pre-trained classifier checkpoint
(`artifacts/tabpfn/checkpoints/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`) inside a
submission archive is not: any such archive must carry `research/tabpfn/LICENSE` and
the attribution line above. Our own fitted artifacts under `artifacts/tabpfn/models/`
hold only the derived feature context and its labels, never redistributed weights.

The `tabpfn-client` package is deliberately never installed or imported; it would send
features to a remote API, which the competition's offline-inference rule forbids.
