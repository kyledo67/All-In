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
