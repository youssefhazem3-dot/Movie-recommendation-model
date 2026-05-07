import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "Movies 67.csv"
BACKUP_PATH = BASE_DIR / "Movies 67_original.csv"
REPORT_PATH = BASE_DIR / "genre_enrichment_report.csv"
TMDB_CACHE_PATH = BASE_DIR / "tmdb_cache.json"

TMDB_GENRES = {
    12: "Adventure",
    14: "Fantasy",
    16: "Animation",
    18: "Drama",
    27: "Horror",
    28: "Action",
    35: "Comedy",
    36: "History",
    37: "Western",
    53: "Thriller",
    80: "Crime",
    99: "Documentary",
    878: "Science Fiction",
    9648: "Mystery",
    10402: "Music",
    10749: "Romance",
    10751: "Family",
    10752: "War",
    10759: "Action,Adventure",
    10762: "Children,Family",
    10763: "News",
    10764: "Reality",
    10765: "Science Fiction,Fantasy",
    10766: "Soap Opera",
    10767: "Talk Show",
    10768: "War,Politics",
    10770: "TV Movie",
}

KNOWN_TITLE_RULES = [
    (r"star trek|star wars|stargate|doctor who|lost in space|outer limits", "Science Fiction,Adventure"),
    (r"twilight zone|x-files", "Science Fiction,Mystery,Thriller"),
    (r"inspector morse|poirot|miss marple|midsomer murders|foyle|cadfael|p\.d\. james|wire in the blood|a touch of frost|tommy & tuppence|campion", "Mystery,Crime,Drama"),
    (r"inspector lynley|prime suspect|dorothy l\.? sayers|lord peter wimsey|agatha christie|ruth rendell|kavanagh q\.?c\.?|baretta|wiseguy|mob justice|mafia|a clean kill|trail of a serial killer|serial killers|unsolved mysteries|scarface|casino:|king of new york|ptu", "Mystery,Crime,Drama"),
    (r"i love lucy|sanford and son|three stooges|monty python|absolutely fabulous|green acres|gilligan|taxi:|king of queens|reno 911|chappelle|black adder|andy griffith|bob hope|carol burnett", "Comedy"),
    (r"duplex|john cleese|designing women|arliss|brady bunch|benny hill|red skelton|mr\.? bean|the lucy show|f troop|dame edna|girls on top|young ones|south park|tom green|george carlin|revenge of the nerds|french & saunders|cedric the entertainer|slapstick|flip wilson|w\.?c\.? fields|laurel & hardy|mr\.? bill|red green|hal roach|mark twain tonight|queer eye|vagina monologues|jeff foxworthy|shrek|jimmy durante|vicar of dibley|ghost and mr\.? chicken|reluctant astronaut|blind date|fat albert", "Comedy"),
    (r"elfen lied", "Animation,Horror,Drama"),
    (r"invader zim|shamanic princess|dominion tank police|soultaker|stellvia|blue seed|chobits|bubblegum crisis|e'?s otherwise|here is greenwood|ramune|saber marionette|boogiepop|ranma|twelve kingdoms|excel saga|angel links|lain|a\.?d\.? police|variable geo|gunsmith cats|neo ranga|kizuna|happy lesson|divergence eve|please twins|shaman king|votoms|chrono crusade|tenchi|dai-guard|sakura diaries|doomed megalopolis|nazca|infinite ryvius|arslan|gravion|geobreeders|witch hunter robin|stratos 4|macross|jubei chan|genshiken|nanako|angelic layer|figure 17|baki|gokusen|dragon half|steel angel kurumi|tsukihime|captain tylor|pretear|princess nine|haibane|agent aika|ad police|all purpose cultural cat girl|shadowraiders|harlock|captain herlock|power rangers|lost universe|wolf'?s rain|midori days|kaleido star|blue submarine|knight hunters|star ocean|please teacher|strawberry eggs|ruin explorers|super gals|my my mai|miyuki-chan", "Animation,Action,Adventure"),
    (r"record of lodoss|slayers|inuyasha|inu-yasha|sailor moon|dragon ball|pokemon|gundam|cowboy bebop|trigun|yu yu hakusho|digimon|transformers|teen titans|spongebob|rugrats|batman the animated", "Animation,Action,Adventure"),
    (r"powerpuff|fairly oddparents|dora the explorer|dragon tales|care bears|franklin|teletubbies|bear in the big blue house|maggie and the ferocious beast|scooby-doo|h\.?r\.? pufnstuf|madeline|arthur|super mario|jimmy neutron|gumby|felix the cat|rainbow fish|dr\.? seuss|charlie brown|beany and cecil|sigmund and the sea monsters|wishbone|lassie|stanley:|american tail|mary-kate|alvin and the chipmunks|read-along|the simpsons|cartoon network|white seal|cricket in times square|frosty|annie:", "Children,Family,Animation"),
    (r"clifford|sesame street|bob the builder|blue'?s clues|thomas & friends|veggietales|baby einstein|caillou|angelina ballerina|disney princess|max and ruby|jay jay the jet plane|rolie polie olie", "Children,Family,Animation"),
    (r"princess diaries|happily ever after|fairy tales|faerie tale|beauty and the beast|narnia|earthsea|mists of avalon|10th kingdom|gormenghast|worst witch|hans christian andersen|frog prince|rumplestiltskin|aladdin", "Family,Fantasy,Adventure"),
    (r"ken burns|national geographic|nova|frontline|american experience|history channel|nature:|travel the world|discovering|walking with|secrets of|unsolved history|mail call|biography", "Documentary"),
    (r"abc primetime|suse orman|suze orman|directors:|making marines|do you believe in miracles|race to freedom|knights templar|gay games|mary magdalene|liberty!|founding fathers|standard deviants|ancient civilizations|modern marvels|beyond the movie|egypt uncovered|everest: imax|titanica: imax|imax|great lodges|ancient egyptians|dolphins|men who killed kennedy|byzantium|jesus and the shroud|d-day|yellowstone: imax|great people of the bible|america beyond|mysteries of the deep|robert thurman|out of ireland|irish empire|meditation and mindfulness|caroline myss|deepak chopra|dalai lama|muslims|auschwitz|1940s house|living planet|city of steel|jeff corwin|extreme engineering|earthquake|nature unleashed|wild america|volcano disaster|nasa|incident at oglala|brakhage|films of charles & ray eames|autopsy files|american steam|portrait of ireland|allergies:|andrew weil|baby human|muhammad ali|celebrities: caught|rome: power|carrier|marihuana|reefer madness", "Documentary"),
    (r"yoga|pilates|tae bo|tai chi|fitness|exercise|denise austin|billy blanks|fat burner|cardio|workout", "Health & Fitness,Exercise"),
    (r"ufc|wwe|wwf|ecw|nba|nfl|nhl|world cup|soccer|pride fc|sports illustrated|stanley cup|boxing|racing|tt 2004|white knuckle extreme|golf|ski movie|nascar|best motoring|motocross|king of the cage|poker|surfing|rollercoasters|fat tire|x games|snowboard|skateboard|sports gone wrong", "Sports"),
    (r"concert|live in|opera|mozart|beethoven|verdi|puccini|rossini|music|musical|band|hits|videos|jazz|symphony|elvis|madonna|britney|metallica|zeppelin|kiss:|billy joel|elton john|bob dylan|dave matthews|depeche mode", "Music"),
    (r"peter tosh|godsmack|jimmy buffett|bizet|carmen|primus|offspring|zakk wylde|ben harper|ozzy|n sync|mtv unplugged|billy bragg|jimi hendrix|korn|tina turner|barry manilow|beatles|radiohead|duran duran|lynyrd skynyrd|oingo boingo|shakira|b\.?b\.? king|jay-z|beach boys|janis joplin|chieftains|carl perkins|eric clapton|stevie ray|david bowie|gilbert and sullivan|mikado|pinafore|pirates of penzance|fourplay|irish tenors|wow gospel|ed sullivan|blues brothers|footloose|nutcracker|swan lake|pas de deux|blank generation|dancin' barefoot|johnny winter|faith hill|eminem|mtv cribs|dave grusin|west side story", "Music"),
    (r"horror|vampire|demon|devil|haunted|zombie|dead|death|fright|creepy|argento|fulci|prom night|house on haunted hill|masque of the red death", "Horror"),
    (r"godzilla|mothra|mimic|mummy|exorcist|nosferatu|jungle holocaust|bram stoker|night stalker|brain damage|monster-a-go-go|psyched by|watchers|nightwalker|ghost stories|eerie, indiana|sweet evil|deranged|i inside|langoliers|doom|witch|blood|clive barker|halfway house|what'?s the matter with helen|whoever slew auntie roo|grossest halloween|fear of a punk planet", "Horror,Thriller"),
    (r"kung fu|shaolin|martial|jackie chan|jet li|bruce lee|samurai|karate|fearless hyena|wong fei hung", "Action,Martial Arts"),
    (r"barbarian queen|davy crockett|tarzan|robin hood|swordsman|sharpe|la femme musketeer|highlander|timerider|bloodfist|bloodsport|best of the best|wing chun|omega doom|blind fury|robocop|charlie'?s angels|secret agent|danger man|avengers '6|airport / airport|airport '77|last run|cross bones|another meltdown|44 minutes|royal deceit|diamond hunters|firefly", "Action,Adventure"),
    (r"western|cowboy|john wayne|roy rogers|mclintock", "Western"),
    (r"war|wwii|world war|military|soldier|churchill|hitler|battle of|combat!|ben-hur|alexander|fidel|castro", "War,History"),
    (r"sex and the city|romance|love|wedding|bride|relationship|danielle steel|red shoe diaries|tales of the city|scenes from a marriage|little house on the prairie|darling buds of may|far pavilions|barchester chronicles|victoria and albert|thomas & sarah|winter people|the cazalets", "Romance,Drama"),
    (r"gupt|lagaan|shahrukh|annahat|waisa bhi hota hai|viruddh|kairee|cilantro y perejil|sexo, pudor y lagrimas|ratas ratones rateros|el coronel|les destinees|drole de drame|fanny trilogy|jean renoir|truffaut|cocteau|renoir|por la libre|tierra|la buche|el grito|5x2", "International,Drama"),
    (r"murder|crime|homicide|law & order|csi|detective|mystery|sherlock", "Crime,Mystery,Drama"),
]

KEYWORD_RULES = [
    (r"documentary|collection|archives|reports", "Documentary"),
    (r"season|series|vol\.|episodes", "TV Show"),
    (r"bonus material|special edition|collector", "Behind The Scenes"),
    (r"comedy|laugh|funny", "Comedy"),
    (r"action|fight|battle", "Action"),
    (r"adventure|journey|quest", "Adventure"),
    (r"drama", "Drama"),
    (r"family|children|kids", "Family"),
    (r"history|historical", "History"),
    (r"science|space|alien|future|sci-fi|scifi", "Science Fiction"),
    (r"fantasy|magic", "Fantasy"),
    (r"thriller|suspense", "Thriller"),
    (r"dance|ballet", "Dance,Music"),
    (r"train|travel|world|africa|china|australia|france", "Travel,Documentary"),
]


def clean_title(title):
    title = str(title).lower()
    title = re.sub(r"\([^)]*\)", " ", title)
    title = re.sub(r"\b(widescreen|full-screen|full screen|limited edition|special edition|collector'?s edition|bonus material|material)\b", " ", title)
    title = re.sub(r"[:\-]+$", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip(" :|-")


def normalize_key(title, year):
    if pd.isna(year):
        return None
    return f"{str(title).strip().lower()}|{int(year)}"


def split_genres(value):
    if value is None or pd.isna(value):
        return []
    parts = []
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in parts:
            parts.append(item)
    return parts


def merge_genres(*genre_values):
    output = []
    for genre_value in genre_values:
        for genre in split_genres(genre_value):
            if genre not in output:
                output.append(genre)
    return ",".join(output)


def tmdb_ids_to_genres(ids):
    if not ids:
        return ""
    return merge_genres(*[TMDB_GENRES.get(int(genre_id), "") for genre_id in ids])


def load_tmdb_cache():
    if not TMDB_CACHE_PATH.exists():
        return {}
    return json.loads(TMDB_CACHE_PATH.read_text(encoding="utf-8"))


def build_existing_genre_lookup(df):
    lookup = {}
    known_rows = df[df["genres"].notna() & (df["genres"].astype(str).str.strip() != "")]

    for row in known_rows.itertuples(index=False):
        key = clean_title(row.title)
        if key and key not in lookup:
            lookup[key] = row.genres

    return lookup


def infer_from_rules(title):
    clean = clean_title(title)

    for pattern, genres in KNOWN_TITLE_RULES:
        if re.search(pattern, clean):
            return genres, "rules"

    for pattern, genres in KEYWORD_RULES:
        if re.search(pattern, clean):
            return genres, "rules"

    return "Drama", "fallback_drama"


def fill_missing_genres():
    if not BACKUP_PATH.exists():
        df = pd.read_csv(MOVIES_PATH)
        df.to_csv(BACKUP_PATH, index=False)
    else:
        df = pd.read_csv(BACKUP_PATH)

    tmdb_cache = load_tmdb_cache()
    existing_lookup = build_existing_genre_lookup(df)
    report_rows = []

    missing_mask = df["genres"].isna() | (df["genres"].astype(str).str.strip() == "")

    for idx, row in df[missing_mask].iterrows():
        title = row["title"]
        year = row["year"]

        source = "rules"
        filled_genres = ""

        tmdb_key = normalize_key(title, year)
        if tmdb_key and tmdb_key in tmdb_cache:
            filled_genres = tmdb_ids_to_genres(tmdb_cache[tmdb_key])
            if filled_genres:
                source = "tmdb_cache"

        if not filled_genres:
            clean = clean_title(title)
            if clean in existing_lookup:
                filled_genres = existing_lookup[clean]
                source = "existing_title_match"

        if not filled_genres:
            filled_genres, source = infer_from_rules(title)

        df.at[idx, "genres"] = filled_genres
        report_rows.append({
            "movie_id": row["movie_id"],
            "title": title,
            "year": year,
            "filled_genres": filled_genres,
            "source": source,
        })

    report_df = pd.DataFrame(report_rows)
    df.to_csv(MOVIES_PATH, index=False)
    report_df.to_csv(REPORT_PATH, index=False)

    print("Original missing genres:", int(missing_mask.sum()))
    print("Remaining missing genres:", int((df["genres"].isna() | (df["genres"].astype(str).str.strip() == "")).sum()))
    print()
    print("Fill sources:")
    print(report_df["source"].value_counts())
    print()
    print("Saved updated file:", MOVIES_PATH)
    print("Saved backup file:", BACKUP_PATH)
    print("Saved report file:", REPORT_PATH)


if __name__ == "__main__":
    fill_missing_genres()
