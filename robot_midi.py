"""
MIDI library for the robot.

Search free MIDI sites, download MIDI files, decode them, and play through the WeDo speaker.

Usage:
    from robot_midi import search_midi, download_midi, play_midi_file, list_songs

    songs = search_midi("fur elise")
    download_midi(songs[0]["url"], "fur_elise.mid")
    play_midi_file(robot, "fur_elise.mid")
"""

import os
import re
import json
import time
import tempfile
import urllib.request
import urllib.parse

try:
    import mido
except ImportError:
    mido = None

MIDI_DIR = os.path.join(tempfile.gettempdir(), "robot_midi")
os.makedirs(MIDI_DIR, exist_ok=True)

# Stop flag and control functions
_midi_stop_requested = False

def stop_midi():
    global _midi_stop_requested
    _midi_stop_requested = True

def clear_midi_stop():
    global _midi_stop_requested
    _midi_stop_requested = False

def get_midi_stop_requested():
    global _midi_stop_requested
    return _midi_stop_requested

# Note name to MIDI number
NOTE_MAP = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

HEADERS = {"User-Agent": "Mozilla/5.0 (RobotBot/1.0)"}


def _midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _note_name_to_midi(name: str) -> int:
    """Convert note name like 'C4' to MIDI number."""
    name = name.strip()
    if len(name) < 2:
        return 60
    letter = name[0].upper()
    accidental = 1 if "#" in name or "b" in name else 0
    octave = int(re.search(r"\d", name).group()) if re.search(r"\d", name) else 4
    note_val = NOTE_MAP.get(letter, 0)
    if "b" in name:
        note_val -= 1
    return octave * 12 + note_val + accidental


# ── Search ─────────────────────────────────────────────────────
def search_midi(query: str, limit: int = 5) -> list[dict]:
    """Search for MIDI files. Returns list of {name, url, id}."""
    # Clean query by replacing symbols and stripping search clogs/stop-words
    q = query.lower().replace("_", " ").replace("-", " ")
    for word in ["song", "music", "theme", "midi", ".mid", "from", "by", "to", "at", "for", "in", "on", "of", "the", "a", "an"]:
        q = re.sub(rf"\b{word}\b", " ", q)
    q = " ".join(q.split())
    if not q:
        q = query

    results = []
    try:
        results = _search_bitmidi(q, limit)
    except Exception as e:
        print(f"BitMIDI search error: {e}", flush=True)

    # Fallback search if first search returned 0 results
    if not results:
        words = q.split()
        if len(words) > 2:
            fallback_query = " ".join(words[:2])
            print(f"Fallback search query: {fallback_query}", flush=True)
            try:
                results = _search_bitmidi(fallback_query, limit)
            except Exception:
                pass

    return results


def _search_bitmidi(query: str, limit: int = 5) -> list[dict]:
    """Search bitmidi.com using their JSON API (same as wario.style)."""
    results = []
    api_url = f"https://bitmidi.com/api/midi/search?q={urllib.parse.quote(query)}"
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(api_url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

            items = data.get("result", {}).get("results", [])
            for item in items[:limit]:
                mid_id = str(item.get("id", "")).strip()
                name = str(item.get("name", "")).strip()
                if not mid_id or not name:
                    continue

                download_url = item.get("downloadUrl", f"/uploads/{mid_id}.mid")
                if not download_url.startswith("http"):
                    download_url = f"https://bitmidi.com{download_url}"

                page_url = item.get("url", "")
                if not page_url and item.get("slug"):
                    page_url = f"https://bitmidi.com/{item['slug']}"
                elif page_url and not page_url.startswith("http"):
                    page_url = f"https://bitmidi.com{page_url}"

                results.append({
                    "name": name,
                    "url": download_url,
                    "page_url": page_url,
                    "id": mid_id,
                    "source": "bitmidi",
                })
            break # Success, exit retry loop
        except Exception as e:
            if attempt == 2:
                print(f"BitMIDI search error: {e}")
            time.sleep(0.5)

    return results


def search_popular(limit: int = 10) -> list[dict]:
    """Get popular MIDI files."""
    try:
        req = urllib.request.Request("https://freemidi.org/topmidi", headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        html = resp.read().decode("utf-8", errors="ignore")

        pattern = r'href="(/download\d+-(\d+)-([^"]+))"[^>]*>([^<]+)<'
        matches = re.findall(pattern, html)

        results = []
        for path, mid_id, slug, title in matches[:limit]:
            results.append({
                "name": title.strip(),
                "url": f"https://freemidi.org{path}",
                "id": mid_id,
            })
        return results
    except Exception:
        return []


# ── Download ───────────────────────────────────────────────────
def download_midi(url: str, filename: str = None) -> str:
    """Download a MIDI file. Returns local path."""
    if filename is None:
        filename = f"midi_{int(time.time())}.mid"
    # sanitize filename
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    filepath = os.path.join(MIDI_DIR, filename)

    # Check cache to avoid repeating downloads
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return filepath

    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=15)
    data = resp.read()

    # check if it's actually HTML (error page)
    if data[:100].decode("utf-8", errors="ignore").strip().startswith("<!"):
        # try to find the actual MIDI link in the page
        html = data.decode("utf-8", errors="ignore")
        midi_match = re.search(r'href="([^"]*\.mid[^"]*)"', html)
        if midi_match:
            midi_url = midi_match.group(1)
            if not midi_url.startswith("http"):
                midi_url = f"https://bitmidi.com{midi_url}"
            req2 = urllib.request.Request(midi_url, headers=HEADERS)
            resp2 = urllib.request.urlopen(req2, timeout=15)
            data = resp2.read()

    with open(filepath, "wb") as f:
        f.write(data)

    return filepath


# ── Helper for Transposing ─────────────────────────────────────
def _transpose_freq(freq: float, semitones: int) -> float:
    """Transpose a frequency by a number of semitones."""
    if semitones == 0:
        return freq
    import math
    try:
        midi_note = round(12.0 * math.log2(freq / 440.0) + 69.0)
        new_note = midi_note + semitones
        return 440.0 * (2.0 ** ((new_note - 69.0) / 12.0))
    except Exception:
        return freq


# ── Parse MIDI ─────────────────────────────────────────────────
def parse_midi(filepath: str, transpose_semitones: int | str = 'auto', play_fraction: float = 0.75,
               min_gap: float = 0.05, min_dur: float = 0.05, monophonic: bool = True) -> dict:
    """Parse a MIDI file and return notes with timing.

    Returns: {
        "tempo": int (bpm),
        "notes": [(freq_hz, duration_sec, velocity, start_time, silence_duration), ...],
        "duration": float (total seconds),
        "tracks": int,
        "name": str
    }
    """
    if mido is None:
        raise ImportError("mido library not installed")

    mid = mido.MidiFile(filepath)
    tempo = 500000  # default: 120 BPM in microseconds per beat
    
    # 1. Parse all notes from all tracks independently to avoid same-pitch overwrites
    notes = []
    for track in mid.tracks:
        current_time = 0.0
        active_notes = {}  # (channel, note) -> list of (start_time, velocity)
        
        for msg in track:
            if msg.time > 0:
                current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
                
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on':
                if msg.velocity > 0:
                    key = (getattr(msg, 'channel', 0), msg.note)
                    if key not in active_notes:
                        active_notes[key] = []
                    active_notes[key].append((current_time, msg.velocity))
                else:
                    key = (getattr(msg, 'channel', 0), msg.note)
                    if key in active_notes and active_notes[key]:
                        start, vel = active_notes[key].pop(0)
                        duration = current_time - start
                        if duration > 0.0001:
                            notes.append({
                                'start': start,
                                'end': current_time,
                                'note': msg.note,
                                'vel': vel
                            })
            elif msg.type == 'note_off':
                key = (getattr(msg, 'channel', 0), msg.note)
                if key in active_notes and active_notes[key]:
                    start, vel = active_notes[key].pop(0)
                    duration = current_time - start
                    if duration > 0.0001:
                        notes.append({
                            'start': start,
                            'end': current_time,
                            'note': msg.note,
                            'vel': vel
                        })
                        
        # Clean up notes left on
        for key, lst in active_notes.items():
            for start, vel in lst:
                notes.append({
                    'start': start,
                    'end': current_time,
                    'note': key[1],
                    'vel': vel
                })
                
    if not notes:
        return {
            "tempo": round(mido.tempo2bpm(tempo)),
            "notes": [],
            "duration": 0.0,
            "tracks": len(mid.tracks),
            "name": os.path.basename(filepath),
        }
        
    notes.sort(key=lambda x: x['start'])
    
    # 2. Monophonic melody extraction via timeline range subtraction
    unmasked_notes = []
    if monophonic:
        for note in notes:
            ranges = [(note['start'], note['end'])]
            for other in notes:
                if other['note'] > note['note'] and other['start'] < note['end'] and note['start'] < other['end']:
                    # Overlap with a higher note! Subtract the higher note's range
                    new_ranges = []
                    for rs, re in ranges:
                        if other['start'] <= rs and other['end'] >= re:
                            # Completely covered
                            continue
                        elif other['start'] > rs and other['end'] < re:
                            # Split
                            new_ranges.append((rs, other['start']))
                            new_ranges.append((other['end'], re))
                        elif other['start'] <= rs and other['end'] > rs:
                            # Truncate start
                            new_ranges.append((other['end'], re))
                        elif other['start'] < re and other['end'] >= re:
                            # Truncate end
                            new_ranges.append((rs, other['start']))
                        else:
                            new_ranges.append((rs, re))
                    ranges = new_ranges
            for rs, re in ranges:
                duration = re - rs
                if duration > 0.001:
                    unmasked_notes.append({
                        'start': rs,
                        'end': re,
                        'note': note['note'],
                        'vel': note['vel']
                    })
    else:
        unmasked_notes = notes
        
    unmasked_notes.sort(key=lambda x: x['start'])
    
    # Resolve 'auto' transposition
    if transpose_semitones == 'auto':
        if unmasked_notes:
            max_note = max(n['note'] for n in unmasked_notes)
            # Target maximum pitch: MIDI note 86 (D6, 1175Hz)
            transpose_semitones = 86 - max_note
        else:
            transpose_semitones = 0
    else:
        transpose_semitones = int(transpose_semitones)
    
    # 3. Apply transpose, staccato spacing, and min duration constraints
    final_notes = []
    for note_info in unmasked_notes:
        start = note_info['start']
        end = note_info['end']
        duration = end - start
        
        note = note_info['note'] + transpose_semitones
        freq = _midi_to_freq(note)
        
        # Calculate play and silence durations
        play_dur = duration * play_fraction
        silence_dur = duration - play_dur
        
        if silence_dur < min_gap:
            play_dur = max(min_dur, duration - min_gap)
            silence_dur = duration - play_dur
            
        if play_dur >= min_dur:
            # We return standard tuple format: (freq, duration, velocity, start, silence_duration)
            final_notes.append((freq, play_dur, note_info['vel'], start, silence_dur))
            
    total_duration = max((n[3] + n[1] for n in final_notes), default=0.0)
    
    return {
        "tempo": round(mido.tempo2bpm(tempo)),
        "notes": final_notes,
        "duration": total_duration,
        "tracks": len(mid.tracks),
        "name": os.path.basename(filepath),
    }


# ── Play through robot ─────────────────────────────────────────
def play_midi_file(robot, filepath: str, speed: float = 1.0, volume: float = 1.0,
                    gap: float = 0.05, min_dur: int = 50, max_dur: int = 600,
                    transpose_semitones: int | str = 'auto', monophonic: bool = True):
    """Play a MIDI file through the robot's speaker.

    Args:
        robot: RobotAPI instance
        filepath: path to .mid file
        speed: playback speed multiplier (1.0 = normal)
        volume: volume multiplier (0.0 to 1.0)
        gap: seconds between notes (default 50ms)
        min_dur: minimum note duration in ms (default 50ms)
        max_dur: maximum note duration in ms (default 600ms)
        transpose_semitones: shift note pitches (default 0)
        monophonic: whether to extract melody only (default True)
    """
    if mido is None:
        raise ImportError("mido library not installed")

    # Use parse_midi to get clean, staccato-spaced notes
    parsed = parse_midi(
        filepath,
        transpose_semitones=transpose_semitones,
        play_fraction=0.75,
        min_gap=gap,
        min_dur=min_dur / 1000.0,
        monophonic=monophonic
    )
    
    play_midi_notes(
        robot,
        parsed,
        speed=speed,
        volume=volume,
        gap=gap,
        min_dur=min_dur,
        max_dur=max_dur,
        transpose_semitones=0  # already transposed in parse_midi
    )


def play_midi_notes(robot, parsed: dict, speed: float = 1.0, volume: float = 1.0,
                     gap: float = 0.05, min_dur: int = 50, max_dur: int = 600,
                     transpose_semitones: int = 0):
    """Play pre-parsed MIDI notes through robot.

    Args:
        robot: RobotAPI instance
        parsed: output from parse_midi()
        speed: playback speed multiplier
        volume: volume multiplier
        gap: seconds between notes
        min_dur: minimum note duration in ms
        max_dur: maximum note duration in ms
        transpose_semitones: shift note pitches
    """
    clear_midi_stop()
    notes = parsed.get("notes", [])
    if not notes:
        return

    # Check if notes have start_time and silence_duration (elements >= 5)
    has_start_time = len(notes[0]) >= 5

    if has_start_time:
        start_playback = time.time()
        for note_tuple in notes:
            if get_midi_stop_requested():
                break
            freq, play_duration, velocity, start_time, silence_duration = note_tuple[:5]
            
            # Apply transposition at play time if specified
            if transpose_semitones != 0:
                freq = _transpose_freq(freq, transpose_semitones)
                
            # Scale duration by volume
            p_dur = play_duration * volume
            
            # Target start time scaled by speed
            target_start = start_playback + (start_time / speed)
            
            now = time.time()
            wait_time = target_start - now
            if wait_time > 0:
                while wait_time > 0:
                    if get_midi_stop_requested():
                        break
                    sleep_chunk = min(0.1, wait_time)
                    time.sleep(sleep_chunk)
                    wait_time -= sleep_chunk
                
            if get_midi_stop_requested():
                break
                
            dur_ms = int(p_dur * 1000)
            if dur_ms > 0:
                dur_ms = max(min_dur, min(dur_ms, max_dur))
                robot.beep(int(freq), dur_ms)
    else:
        # Fallback to simple sequential play if notes are in old format
        for note_tuple in notes:
            if get_midi_stop_requested():
                break
            freq = note_tuple[0]
            duration = note_tuple[1]
            
            if transpose_semitones != 0:
                freq = _transpose_freq(freq, transpose_semitones)
                
            dur_ms = int(duration * 1000 * volume / speed)
            dur_ms = max(min_dur, min(dur_ms, max_dur))
            robot.beep(int(freq), dur_ms)
            
            wait_time = gap / speed
            while wait_time > 0:
                if get_midi_stop_requested():
                    break
                sleep_chunk = min(0.1, wait_time)
                time.sleep(sleep_chunk)
                wait_time -= sleep_chunk


def preview_midi(filepath: str) -> str:
    """Get a text summary of a MIDI file."""
    if mido is None:
        return "mido not installed"

    parsed = parse_midi(filepath)
    lines = [
        f"File: {parsed['name']}",
        f"Tracks: {parsed['tracks']}",
        f"Tempo: {parsed['tempo']} BPM",
        f"Duration: {parsed['duration']:.1f}s",
        f"Notes: {len(parsed['notes'])}",
    ]

    if parsed["notes"]:
        freqs = [n[0] for n in parsed["notes"]]
        lines.append(f"Range: {min(freqs):.0f}Hz - {max(freqs):.0f}Hz")

    return "\n".join(lines)


# ── Utilities ──────────────────────────────────────────────────
def list_songs() -> list[str]:
    """List downloaded MIDI files."""
    if not os.path.exists(MIDI_DIR):
        return []
    return [f for f in os.listdir(MIDI_DIR) if f.endswith(".mid")]


def get_song_path(filename: str) -> str:
    """Get full path to a downloaded song."""
    return os.path.join(MIDI_DIR, filename)


def delete_song(filename: str) -> str:
    """Delete a downloaded MIDI file."""
    path = os.path.join(MIDI_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return f"Deleted {filename}"
    return f"{filename} not found"


def search_and_play(robot, query: str, speed: float = 1.0) -> str:
    """Search, download, and play the first result. Returns status message."""
    results = search_midi(query, limit=1)
    if not results:
        return f"No MIDI found for '{query}'"

    song = results[0]
    filepath = download_midi(song["url"], f"{song['name']}.mid")
    play_midi_file(robot, filepath, speed=speed)
    return f"Played: {song['name']}"


def demo_scale(robot):
    """Play a demo scale to test speaker."""
    notes = [262, 294, 330, 349, 392, 440, 494, 523]
    for freq in notes:
        robot.beep(freq, 200)
        time.sleep(0.05)
