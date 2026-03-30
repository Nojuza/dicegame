from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import random
from itertools import product

app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:4000',
    'http://127.0.0.1:4000',
    'https://*.pages.dev',
])

# ─── Hand Classification ───────────────────────────────────────────

def classify_roll(dice, num_sides):
    freq = {}
    for d in dice:
        freq[d] = freq.get(d, 0) + 1
    max_kind = max(freq.values()) if freq else 0

    present = set(dice)
    longest_run = 1 if present else 0
    run = 1
    for v in range(2, num_sides + 1):
        if v in present and (v - 1) in present:
            run += 1
            if run > longest_run:
                longest_run = run
        elif v in present:
            run = 1
    if len(present) <= 1:
        longest_run = len(present)

    return max_kind, longest_run, freq, present


def score_hand(dice, config):
    max_kind, longest_run, freq, present = classify_roll(dice, config['sides'])
    ranking = config['ranking']
    disabled = set(config.get('disabledHands', []))
    kind_sub = config.get('kindSubValues', {})

    best_rank = -1
    best_key = None
    for i, key in enumerate(ranking):
        if key in disabled:
            continue
        typ, n_str = key.split('-')
        n = int(n_str)
        qualifies = False
        if typ == 'straight' and longest_run >= n:
            qualifies = True
        if typ == 'kind' and max_kind >= n:
            qualifies = True
        if qualifies:
            best_rank = len(ranking) - i
            best_key = key
            break

    sub_value = 0
    if best_key:
        typ, n_str = best_key.split('-')
        n = int(n_str)
        if typ == 'kind':
            sub_order = kind_sub.get(best_key, [])
            for val_str, cnt in freq.items():
                val = int(val_str) if isinstance(val_str, str) else val_str
                if cnt >= n:
                    try:
                        idx = sub_order.index(val)
                        sv = len(sub_order) - idx
                    except (ValueError, IndexError):
                        sv = 0
                    if sv > sub_value:
                        sub_value = sv
        else:
            for start in range(config['sides'], 0, -1):
                r = 0
                for v in range(start, 0, -1):
                    if v in present:
                        r += 1
                    else:
                        break
                if r >= n:
                    sub_value = start
                    break

    # Second-best hand (other category)
    second_rank = -1
    second_sub = 0
    if best_key:
        best_type = best_key.split('-')[0]
        for i, key in enumerate(ranking):
            if key == best_key or key in disabled:
                continue
            typ, n_str = key.split('-')
            n = int(n_str)
            if typ == best_type:
                continue
            qualifies = False
            if typ == 'straight' and longest_run >= n:
                qualifies = True
            if typ == 'kind' and max_kind >= n:
                qualifies = True
            if qualifies:
                second_rank = len(ranking) - i
                if typ == 'kind':
                    sub_order = kind_sub.get(key, [])
                    for val_str, cnt in freq.items():
                        val = int(val_str) if isinstance(val_str, str) else val_str
                        if cnt >= n:
                            try:
                                idx = sub_order.index(val)
                                sv = len(sub_order) - idx
                            except (ValueError, IndexError):
                                sv = 0
                            if sv > second_sub:
                                second_sub = sv
                else:
                    for start in range(config['sides'], 0, -1):
                        r = 0
                        for v in range(start, 0, -1):
                            if v in present:
                                r += 1
                            else:
                                break
                        if r >= n:
                            second_sub = start
                            break
                break

    return (best_rank, sub_value, second_rank, second_sub)


def compare_hands(d1, d2, config):
    s1 = score_hand(d1, config)
    s2 = score_hand(d2, config)
    for a, b in zip(s1, s2):
        if a > b:
            return 1
        if a < b:
            return -1
    return 0


# ─── Probability Calculation ───────────────────────────────────────

def calculate_probabilities(num_dice, num_sides, min_straight):
    total = num_sides ** num_dice
    ms = min_straight
    max_straight = min(num_dice, num_sides)

    straight_counts = {}
    kind_counts = {}
    specific_kind = {}
    specific_straight = {}

    use_enum = total <= 600000
    sample_count = total if use_enum else 500000

    if use_enum:
        for combo in product(range(1, num_sides + 1), repeat=num_dice):
            dice = list(combo)
            _tally(dice, num_sides, ms, max_straight, num_dice,
                   straight_counts, kind_counts, specific_kind, specific_straight)
    else:
        for _ in range(sample_count):
            dice = [random.randint(1, num_sides) for _ in range(num_dice)]
            _tally(dice, num_sides, ms, max_straight, num_dice,
                   straight_counts, kind_counts, specific_kind, specific_straight)

    return {
        'straightCounts': straight_counts,
        'kindCounts': kind_counts,
        'specificKind': specific_kind,
        'specificStraight': specific_straight,
        'total': sample_count,
        'simulated': not use_enum,
        'maxStraight': max_straight,
        'ms': ms,
    }


def _tally(dice, num_sides, ms, max_straight, num_dice,
           straight_counts, kind_counts, specific_kind, specific_straight):
    max_kind, longest_run, freq, present = classify_roll(dice, num_sides)

    if longest_run >= ms:
        sk = f'straight-{longest_run}'
        straight_counts[sk] = straight_counts.get(sk, 0) + 1
    else:
        straight_counts['no-straight'] = straight_counts.get('no-straight', 0) + 1

    if max_kind >= 2:
        kk = f'kind-{max_kind}'
        kind_counts[kk] = kind_counts.get(kk, 0) + 1
    else:
        kind_counts['no-kind'] = kind_counts.get('no-kind', 0) + 1

    freq1 = freq.get(1, 0)
    for n in range(2, num_dice + 1):
        if freq1 >= n:
            sk = f'kind-{n}'
            specific_kind[sk] = specific_kind.get(sk, 0) + 1

    for n in range(ms, max_straight + 1):
        if all(v in present for v in range(1, n + 1)):
            sk = f'straight-{n}'
            specific_straight[sk] = specific_straight.get(sk, 0) + 1


# ─── Hand Strength Estimation ─────────────────────────────────────

def estimate_strength(known_personal, known_dealer, config, samples=100):
    sides = config['sides']
    unknown_personal = 3 - len(known_personal)
    unknown_dealer = 2 - len(known_dealer)
    wins = 0.0

    for _ in range(samples):
        my_hand = list(known_personal)
        for _ in range(unknown_personal):
            my_hand.append(random.randint(1, sides))
        dealer = list(known_dealer)
        for _ in range(unknown_dealer):
            dealer.append(random.randint(1, sides))
        my_full = my_hand + dealer

        opp_hand = [random.randint(1, sides) for _ in range(3)]
        opp_full = opp_hand + dealer

        cmp = compare_hands(my_full, opp_full, config)
        if cmp > 0:
            wins += 1
        elif cmp == 0:
            wins += 0.5

    return wins / samples


# ─── AI & Bluff Logic ─────────────────────────────────────────────

AI_PROFILE_DEFAULTS = {
    'TP': {'foldBelow': 0.5, 'raiseAbove': 0.8, 'checkBluff': 0.02, 'betBluff': 0.05, 'raiseMultiplier': 2},
    'TA': {'foldBelow': 0.4, 'raiseAbove': 0.55, 'checkBluff': 0.03, 'betBluff': 0.05, 'raiseMultiplier': 2},
    'LP': {'foldBelow': 0.15, 'raiseAbove': 0.7, 'checkBluff': 0.08, 'betBluff': 0.12, 'raiseMultiplier': 2},
    'LA': {'foldBelow': 0.2, 'raiseAbove': 0.4, 'checkBluff': 0.15, 'betBluff': 0.20, 'raiseMultiplier': 3},
}


def get_ai_action(strategy, strength, to_call, can_check, big_blind, chips, current_bet, profiles=None):
    all_profiles = profiles if profiles else AI_PROFILE_DEFAULTS
    prof = all_profiles.get(strategy, all_profiles.get('TA', AI_PROFILE_DEFAULTS['TA']))

    if strength < prof['foldBelow']:
        if can_check:
            if random.random() < prof['checkBluff']:
                raise_amt = current_bet + big_blind * prof['raiseMultiplier']
                return 'raise', min(raise_amt, chips + current_bet), True, False
            return 'check', 0, False, False
        # Probabilistic bluff call: call with probability = strength
        if random.random() < strength:
            return 'call', 0, False, True
        if random.random() < prof['betBluff']:
            raise_amt = current_bet + big_blind * prof['raiseMultiplier']
            return 'raise', min(raise_amt, chips + current_bet), True, False
        return 'fold', 0, False, False

    if strength >= prof['raiseAbove']:
        raise_amt = current_bet + big_blind * prof['raiseMultiplier']
        return 'raise', min(raise_amt, chips + current_bet), False, False

    if can_check:
        return 'check', 0, False, False
    return 'call', 0, False, False


# ─── Batch Simulation ─────────────────────────────────────────────

def simulate_batch(config):
    num_games = config['numGames']
    num_p = config['numPlayers']
    strategies = config['strategies'][:num_p]
    start_chips = config['startChips']
    sb_amt = config['smallBlind']
    bb_amt = config['bigBlind']
    sides = config['sides']
    hand_config = {
        'sides': sides,
        'minStraight': config['minStraight'],
        'ranking': config['ranking'],
        'kindSubValues': config.get('kindSubValues', {}),
        'disabledHands': config.get('disabledHands', []),
    }
    ai_profiles = config.get('aiProfiles', None)

    # Stats tracking
    wins = [0] * num_p
    chip_change = [0.0] * num_p
    hand_dist = {}
    fold_count = [0] * num_p
    total_actions = [0] * num_p
    bluff_raises = [0] * num_p
    bluff_wins = [0] * num_p
    bluffs_called = [0] * num_p
    bluff_call_attempts = [0] * num_p
    bluff_call_correct = [0] * num_p
    total_showdowns = 0
    total_showdown_hands = 0

    for g in range(num_games):
        # Initialize players
        players = []
        for i in range(num_p):
            s = strategies[i] if strategies[i] != 'human' else 'TA'
            players.append({
                'strategy': s,
                'chips': start_chips,
                'dice': [0, 0, 0],
                'folded': False,
                'current_bet': 0,
                'total_bet': 0,
                'all_in': False,
                'acted': False,
                'last_bluff': False,  # did their last raise = bluff?
                'made_bluff_call': False,
            })

        pot = 0
        current_bet = 0
        dealer_pos = g % num_p
        dealer_dice = [0, 0]

        # Blinds
        sb_idx = (dealer_pos + 1) % num_p
        bb_idx = (dealer_pos + 2) % num_p
        sb_actual = min(sb_amt, players[sb_idx]['chips'])
        players[sb_idx]['chips'] -= sb_actual
        players[sb_idx]['current_bet'] = sb_actual
        players[sb_idx]['total_bet'] = sb_actual
        pot += sb_actual

        bb_actual = min(bb_amt, players[bb_idx]['chips'])
        players[bb_idx]['chips'] -= bb_actual
        players[bb_idx]['current_bet'] = bb_actual
        players[bb_idx]['total_bet'] = bb_actual
        pot += bb_actual
        current_bet = bb_actual

        game_done = False

        for rnd in range(1, 4):
            if game_done:
                break

            # Roll dice
            for p in players:
                if not p['folded']:
                    p['dice'][rnd - 1] = random.randint(1, sides)
            if rnd >= 2:
                dealer_dice[rnd - 2] = random.randint(1, sides)

            # Betting round
            if rnd > 1:
                current_bet = 0
            for p in players:
                p['current_bet'] = 0
                p['acted'] = False

            start_p = (dealer_pos + 1) % num_p
            max_iterations = num_p * 6

            for _ in range(max_iterations):
                active = [p for p in players if not p['folded'] and not p['all_in']]
                if len(active) <= 0:
                    break
                if all(p['acted'] and p['current_bet'] >= current_bet for p in active):
                    break

                for step in range(num_p):
                    idx = (start_p + step) % num_p
                    p = players[idx]
                    if p['folded'] or p['all_in']:
                        continue
                    if p['acted'] and p['current_bet'] >= current_bet:
                        continue

                    total_actions[idx] += 1
                    known_personal = p['dice'][:rnd]
                    known_dealer = [d for d in dealer_dice if d > 0]
                    strength = estimate_strength(known_personal, known_dealer,
                                                 hand_config, samples=50)

                    to_call = current_bet - p['current_bet']
                    can_check = to_call <= 0

                    action, amount, is_bluff, is_bluff_call = get_ai_action(
                        p['strategy'], strength, to_call, can_check,
                        bb_amt, p['chips'], p['current_bet'], ai_profiles
                    )

                    if is_bluff:
                        bluff_raises[idx] += 1
                        p['last_bluff'] = True
                    elif action == 'raise':
                        p['last_bluff'] = False

                    if is_bluff_call:
                        bluff_call_attempts[idx] += 1
                        p['made_bluff_call'] = True

                    if action == 'fold':
                        p['folded'] = True
                        fold_count[idx] += 1
                    elif action == 'check':
                        pass
                    elif action == 'call':
                        to_add = min(to_call, p['chips'])
                        p['chips'] -= to_add
                        p['current_bet'] += to_add
                        p['total_bet'] += to_add
                        pot += to_add
                        if p['chips'] == 0:
                            p['all_in'] = True
                        # Check if calling a bluffer
                        raiser_idx = _find_last_raiser(players, idx)
                        if raiser_idx >= 0 and players[raiser_idx]['last_bluff']:
                            bluffs_called[raiser_idx] += 1
                    elif action == 'raise':
                        to_add = min(amount - p['current_bet'], p['chips'])
                        p['chips'] -= to_add
                        p['current_bet'] += to_add
                        p['total_bet'] += to_add
                        pot += to_add
                        current_bet = p['current_bet']
                        if p['chips'] == 0:
                            p['all_in'] = True
                        for oi, op in enumerate(players):
                            if oi != idx and not op['folded'] and not op['all_in']:
                                op['acted'] = False

                    p['acted'] = True

                # Re-check completion
                active = [p for p in players if not p['folded'] and not p['all_in']]
                if len(active) <= 0 or all(
                    p['acted'] and p['current_bet'] >= current_bet for p in active
                ):
                    break

            remaining = [(i, p) for i, p in enumerate(players) if not p['folded']]
            if len(remaining) <= 1:
                game_done = True
                break

        # Showdown
        remaining = [(i, p) for i, p in enumerate(players) if not p['folded']]
        if len(remaining) == 1:
            winner_idx = remaining[0][0]
            remaining[0][1]['chips'] += pot
            wins[winner_idx] += 1
            # If the winner was bluffing, it's a bluff win
            if players[winner_idx]['last_bluff']:
                bluff_wins[winner_idx] += 1
        else:
            total_showdowns += 1
            total_showdown_hands += len(remaining)
            best_hand = None
            best_players = []

            for idx, p in remaining:
                full_hand = p['dice'] + dealer_dice
                # Track hand distribution
                mk, lr, _, _ = classify_roll(full_hand, sides)
                ms = hand_config['minStraight']
                if lr >= ms:
                    hk = f'straight-{lr}'
                    hand_dist[hk] = hand_dist.get(hk, 0) + 1
                if mk >= 2:
                    hk = f'kind-{mk}'
                    hand_dist[hk] = hand_dist.get(hk, 0) + 1

                if best_hand is None:
                    best_hand = full_hand
                    best_players = [(idx, p)]
                else:
                    cmp = compare_hands(full_hand, best_hand, hand_config)
                    if cmp > 0:
                        best_hand = full_hand
                        best_players = [(idx, p)]
                    elif cmp == 0:
                        best_players.append((idx, p))

            share = pot // len(best_players)
            win_share = 1.0 / len(best_players)
            for bi, (idx, p) in enumerate(best_players):
                p['chips'] += share + (pot % len(best_players) if bi == 0 else 0)
                wins[idx] += win_share

            # Check bluff call correctness at showdown:
            # For each player who bluff-called, check if any bluffer lost
            best_idxs = set(bi for bi, _ in best_players)
            for idx, p in remaining:
                if p.get('made_bluff_call'):
                    # Was there a bluffer who lost?
                    for oi, op in remaining:
                        if oi != idx and op.get('last_bluff') and oi not in best_idxs:
                            bluff_call_correct[idx] += 1
                            break  # only count once per showdown

        for i, p in enumerate(players):
            chip_change[i] += (p['chips'] - start_chips)

    # Build response
    bluff_stats = []
    for i in range(num_p):
        br = bluff_raises[i]
        ta = total_actions[i]
        bluff_stats.append({
            'bluffRate': br / ta if ta > 0 else 0,
            'bluffSuccess': bluff_wins[i] / br if br > 0 else 0,
            'bluffRaises': br,
            'bluffsCalled': bluffs_called[i],
            'bluffCallAttempts': bluff_call_attempts[i],
            'bluffCallCorrect': bluff_call_correct[i] / bluff_call_attempts[i]
                if bluff_call_attempts[i] > 0 else 0,
        })

    return {
        'winRates': [w / num_games for w in wins],
        'avgChipChange': [c / num_games for c in chip_change],
        'handDistribution': hand_dist,
        'totalShowdownHands': total_showdown_hands,
        'bluffStats': bluff_stats,
        'foldRates': [f / (ta if ta > 0 else 1)
                      for f, ta in zip(fold_count, total_actions)],
    }


def _find_last_raiser(players, exclude_idx):
    """Find the index of the last player who raised (not the current player)."""
    for i, p in enumerate(players):
        if i != exclude_idx and not p['folded'] and p['current_bet'] > 0 and p.get('last_bluff'):
            return i
    return -1


# ─── API Routes ────────────────────────────────────────────────────

@app.route('/api/calculate-probabilities', methods=['POST'])
def api_calc_probs():
    data = request.get_json()
    result = calculate_probabilities(
        data.get('numDice', 5),
        data.get('numSides', 6),
        data.get('minStraight', 3),
    )
    return jsonify(result)


@app.route('/api/batch-simulate', methods=['POST'])
def api_batch_sim():
    data = request.get_json()
    result = simulate_batch(data)
    return jsonify(result)


@app.route('/api/hand-strength', methods=['POST'])
def api_hand_strength():
    data = request.get_json()
    config = {
        'sides': data.get('sides', 6),
        'minStraight': data.get('minStraight', 3),
        'ranking': data.get('ranking', []),
        'kindSubValues': data.get('kindSubValues', {}),
        'disabledHands': data.get('disabledHands', []),
    }
    strength = estimate_strength(
        data.get('knownPersonal', []),
        data.get('knownDealer', []),
        config,
        samples=data.get('numSamples', 100),
    )
    return jsonify({'strength': strength})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print('DiceHands backend running at http://localhost:5000')
    app.run(port=5000, debug=False)
