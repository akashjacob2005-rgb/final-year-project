"""Build the proper-noun blocklist used by verbal-fluency scoring.

Phonemic fluency (the FAS task) excludes proper nouns: "no names of people or
places" is part of the instruction the participant is read. Scoring has to
enforce that, and the wordhood check in ml/lexicon.py cannot — it is built from
WordNet, which lists France and Florida as nouns exactly like fox and flower.

Output: ml/artifacts/proper_nouns.txt, one lowercased entry per line, sorted.
That file is COMMITTED. Do not make the app generate it at runtime: the BSD
dictionaries this reads from exist on macOS and BSD but not in the Linux
container the backend deploys to.

    python ml/build_proper_nouns.py

Sources, and why each is here
-----------------------------
1. /usr/share/dict/web2 — Webster's Second International (1934), public domain,
   shipped with macOS and the BSDs. Entries are capitalised iff Webster's
   treated them as proper. We keep a word only when it appears capitalised and
   NEVER lowercase, which is the whole safeguard against over-blocking: "frank"
   (candid) and "august" (majestic) are lowercase entries too, so they survive
   and stay scoreable, while "Florida" and "Sunday" do not.

2. /usr/share/dict/propernames — the BSD list of common given names.

3. GAZETTEER below — hand-written, because Webster's 1934 is a dictionary and
   not an atlas: it has no entry at all for France, Spain, Africa or Scotland,
   which are precisely the answers a participant reaches for. Countries,
   continents, oceans, UK nations, US states and large world cities.

Deliberately NOT included: demonyms and language names (french, spanish,
swedish). They are proper-noun derived and a strict examiner would reject them,
but several are also ordinary lowercase words, and wrongly rejecting a word the
participant genuinely produced costs more than leniently accepting one.
"""

from __future__ import annotations

from pathlib import Path

WEB2 = Path("/usr/share/dict/web2")
PROPERNAMES = Path("/usr/share/dict/propernames")
OUT = Path(__file__).resolve().parent / "artifacts" / "proper_nouns.txt"

GAZETTEER = """
afghanistan albania algeria andorra angola antarctica antigua argentina armenia
asia australia austria azerbaijan bahamas bahrain bangladesh barbados belarus
belgium belize benin bhutan bolivia bosnia botswana brazil britain brunei
bulgaria burkina burundi cambodia cameroon canada chad chile china colombia
comoros congo croatia cuba cyprus czechia denmark djibouti dominica ecuador
egypt eritrea estonia eswatini ethiopia europe fiji finland france gabon gambia
georgia germany ghana greece greenland grenada guatemala guinea guyana haiti
honduras hungary iceland india indonesia iran iraq ireland israel italy jamaica
japan jordan kazakhstan kenya kiribati kosovo kuwait kyrgyzstan laos latvia
lebanon lesotho liberia libya liechtenstein lithuania luxembourg madagascar
malawi malaysia maldives mali malta mauritania mauritius mexico micronesia
moldova monaco mongolia montenegro morocco mozambique myanmar namibia nauru
nepal netherlands nicaragua niger nigeria norway oman pakistan palau palestine
panama paraguay peru philippines poland portugal qatar romania russia rwanda
samoa scotland senegal serbia seychelles singapore slovakia slovenia somalia
spain sudan suriname sweden switzerland syria taiwan tajikistan tanzania
thailand togo tonga tunisia turkey turkmenistan tuvalu uganda ukraine uruguay
uzbekistan vanuatu vatican venezuela vietnam wales yemen zambia zimbabwe
africa america antarctica arctic atlantic pacific mediterranean caribbean
himalaya himalayas amazon danube nile thames
alabama alaska arizona arkansas california colorado connecticut delaware
florida hawaii idaho illinois indiana iowa kansas kentucky louisiana maine
maryland massachusetts michigan minnesota mississippi missouri montana nebraska
nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia
washington wisconsin wyoming
amsterdam athens atlanta baghdad bangalore bangkok barcelona beijing beirut
belfast berlin birmingham bogota boston brisbane brussels bucharest budapest
buenos cairo calcutta cardiff casablanca chennai chicago copenhagen dallas
delhi detroit dhaka dubai dublin durban edinburgh frankfurt geneva glasgow
hamburg hanoi havana helsinki houston istanbul jakarta jerusalem johannesburg
karachi kolkata lagos lahore lima lisbon liverpool london madrid manchester
manila melbourne miami milan montreal moscow mumbai munich nairobi naples
newcastle orleans oslo ottawa paris perth philadelphia phoenix prague pune
quebec riyadh rome santiago seattle seoul shanghai sheffield singapore sofia
stockholm sydney taipei tehran tokyo toronto valencia vancouver venice vienna
warsaw wellington yokohama zurich
"""


def _read(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. This script needs the BSD dictionaries, so run "
            "it on macOS or a BSD; the committed output is what ships."
        )
    # latin-1: web2 carries a few accented headwords and never fails to decode.
    return path.read_text(encoding="latin-1").splitlines()


def build() -> set[str]:
    lowercase: set[str] = set()
    capitalised: set[str] = set()
    for line in _read(WEB2):
        word = line.strip()
        if not word.isascii() or not word.isalpha() or len(word) < 2:
            continue
        (lowercase if word.islower() else capitalised).add(word.lower())

    given_names = {
        line.strip().lower()
        for line in _read(PROPERNAMES)
        if line.strip().isalpha() and len(line.strip()) >= 2
    }

    gazetteer = {w for w in GAZETTEER.split() if len(w) >= 2}

    # The subtraction is the safeguard: anything Webster's also knows as an
    # ordinary lowercase word is never blocked, whatever the other sources say.
    return (capitalised | given_names | gazetteer) - lowercase


def main() -> None:
    words = sorted(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"wrote {len(words)} entries to {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
