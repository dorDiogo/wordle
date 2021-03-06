// #define NDEBUG
#include <bits/stdc++.h>
#include <omp.h>

#define dbg(x) cerr << ">>> " << x << endl;
#define _ << " | " <<

using namespace std;

const int kMaxAllowedGuesses = 6;

const int kInf = int(1e8);

const int kAnswersDictSize = 2400;
const int kWordsDictSize = 13000;
const int kWordSize = 5;

const int kGreenMatch = 0, kYellowMatch = 1, kBlackMatch = 2;

const int kExactMatch = 0;
const int kMaxEncodedPattern = []() {
  int max_encoded_pattern = 0;
  for (int i = 0; i < kWordSize; ++i) {
    max_encoded_pattern = 3 * max_encoded_pattern + 2;
  }
  return max_encoded_pattern;
}();

string answers[kAnswersDictSize];
string dict[kWordsDictSize];
int answers_size = 0, dict_size = 0;

map<vector<short>, pair<int, int>> memo[kMaxAllowedGuesses][kWordsDictSize][kMaxEncodedPattern + 1];
map<vector<short>, int> memo_upper[kMaxAllowedGuesses][kWordsDictSize][kMaxEncodedPattern + 1];
int memo_upper_size = 0;
int memo_size = 0;
int bucket_sizes[kMaxEncodedPattern + 1] = {0};
long long need_reset[kMaxEncodedPattern + 1] = {0};
long long iteration = 0;
unsigned char pattern_matrix[kWordsDictSize][kWordsDictSize];

// Reads a word from an input file that has been masked by shifting the ascii char value by +5.
string ReadWord(ifstream& input) {
  string a;
  input >> a;
  for (auto& ch : a) {
    ch -= 5;
    ch = (char)tolower(ch);
    assert(ch >= 'a' && ch <= 'z');
  }
  assert((int)a.size() == kWordSize);
  return a;
}

// Reads input file to answers and dict arrays.
void ReadGameDict(const string& game) {
  ifstream input(game);
  cerr << game << endl;

  // Read possible answers.
  input >> answers_size;
  set<string> set_answers;
  for (int i = 0; i < answers_size; ++i) {
    set_answers.insert(ReadWord(input));
  }
  assert((int)set_answers.size() < kAnswersDictSize);

  // Read rest of dictionary.
  input >> dict_size;
  set<string> set_dict;
  for (int i = 0; i < dict_size; ++i) {
    set_dict.insert(ReadWord(input));
  }
  input.close();

  answers_size = 0;
  for (const string& answer : set_answers) {
    answers[answers_size++] = answer;
  }

  set_dict.insert(set_answers.begin(), set_answers.end());
  assert((int)set_dict.size() < kWordsDictSize);
  dict_size = 0;
  for (const string& word : set_dict) {
    dict[dict_size++] = word;
  }

  cerr << "Dictionary size: " << dict_size << endl;
  cerr << "Possible answers size: " << answers_size << endl;
}

void SortDictWordsByLetterFrequency() {
  map<char, int> letter_frequency;
  for (int i = 0; i < answers_size; ++i) {
    for (char letter : answers[i]) {
      ++letter_frequency[letter];
    }
  }

  auto word_letter_frequency = [&](const string& word) {
    int frequency = 0;
    for (char letter : set<char>(word.begin(), word.end())) {
      frequency += letter_frequency[letter];
    }
    return frequency;
  };

  sort(dict, dict + dict_size,
       [&](const string& lhs, const string& rhs) { return word_letter_frequency(lhs) > word_letter_frequency(rhs); });
}

string DecodePattern(int v, bool emoji = false) {
  string a;
  while (v) {
    a += char((v % 3) + int('0'));
    v /= 3;
  }
  while (a.size() < 5) {
    a += '0';
  }
  string b;
  reverse(a.begin(), a.end());
  for (auto ch : a) {
    if (ch == '2') {
      b += emoji ? "\u2B1B" : "B";
    } else if (ch == '0') {
      b += emoji ? u8"\U0001F7E9" : "G";
    } else if (ch == '1') {
      b += emoji ? u8"\U0001F7E8" : "Y";
    }
  }
  return b;
}

// Computes the resulting pattern for the corresponding guess/answer pair.
// Encoded as a base-3 number where 0 = GREEN, 1 = YELLOW, 2 = BLACK.
// E.g., ???????????????????????? = 2 * 3^4 + 1 * 3^3 + 2 * 3^2 + 1 * 3^1 + 0 * 3^0.
int ComputePattern(const string& guess, const string& answer) {
  vector<bool> matched(kWordSize, false);
  vector<int> pattern(kWordSize, kBlackMatch);

  // Check for letters in the correct spot.
  for (int i = 0; i < kWordSize; ++i) {
    if (guess[i] == answer[i]) {
      pattern[i] = kGreenMatch;
      matched[i] = true;
    }
  }

  // Check for letters in the wrong spot.
  for (int i = 0; i < kWordSize; ++i) {
    for (int j = 0; j < kWordSize && pattern[i] == kBlackMatch; ++j) {
      if (matched[j]) continue;
      if (guess[i] == answer[j]) {
        pattern[i] = kYellowMatch;
        matched[j] = true;
      }
    }
  }

  // Encode pattern as a base-3 number.
  int encoded_pattern = 0;
  for (int match : pattern) {
    encoded_pattern = int(3) * encoded_pattern + match;
  }
  return encoded_pattern;
}

// Computes pattern matrix where pattern_matrix[i][j] = compute_pattern(dict[i], dict[j]).
void ComputePatternMatrix() {
#pragma omp parallel for
  for (int i = 0; i < dict_size; ++i) {
    for (int j = 0; j < dict_size; ++j) {
      pattern_matrix[i][j] = (unsigned char)ComputePattern(dict[i], dict[j]);
    }
  }
}

bool IsHardModeCompatible(int previous_guess, int previous_pattern, int guess) {
  string decoded_previous_pattern = DecodePattern(previous_pattern);
  string decoded_pattern = DecodePattern(pattern_matrix[previous_guess][guess]);
  for (int i = 0; i < (int)decoded_pattern.size(); ++i) {
    if (decoded_previous_pattern[i] == 'G' && decoded_pattern[i] != 'G') {
      return false;
    } else if (decoded_previous_pattern[i] == 'Y' && decoded_pattern[i] == 'B') {
      return false;
    }
  }
  return true;
}

// Computes the set of dictionary words minus the set of remaining answers.
vector<short> DictMinusRemainingAnswers(const vector<short>& remaining_answers) {
  vector<bool> is_not_answer(dict_size, true);
  vector<short> not_answers;
  for (short answer : remaining_answers) {
    is_not_answer[answer] = false;
  }
  for (short i = 0; i < dict_size; ++i) {
    if (is_not_answer[i]) {
      not_answers.push_back(i);
    }
  }
  return not_answers;
}

// Solves wordle by doing a complete search in its guess+pattern tree.
int Dfs(const int initial_guess, const vector<short>& remaining_answers, const int parent_score_upper_bound = kInf,
        const int depth = 1, int previous_guess = -1, int previous_pattern = -1) {
  if (depth > kMaxAllowedGuesses) {
    return kInf;
  }
  if ((int)remaining_answers.size() == 1) {
    return 1;
  }
  if (depth + 1 > kMaxAllowedGuesses) {
    return kInf;
  }

  auto& _memo = memo[depth - 1][previous_guess][previous_pattern];
  auto& _memo_upper = memo_upper[depth - 1][previous_guess][previous_pattern];

  // Score is the sum how many guesses are needed to find the answer for all remaining answers.
  int best_score = kInf;
  int best_guess = 0;
  auto entry = _memo.find(remaining_answers);
  if (entry != _memo.end()) {
    return entry->second.first;
  }
  auto entry_upper = _memo_upper.find(remaining_answers);
  if (entry_upper != _memo_upper.end()) {
    if (parent_score_upper_bound <= entry_upper->second) {
      return kInf;
    }
  }

  vector<short> not_answers;
  set<short> set_remaining_answers;
  if (depth == 1) {
    set_remaining_answers = set<short>(remaining_answers.begin(), remaining_answers.end());
  }

  for (short aux = 0; aux < dict_size; ++aux) {
    if (depth == 1 && initial_guess != aux) {
      continue;
    }
    short guess;

    // Speed-up: process remaining answers first, and then the rest of the dictionary.
    if (depth == 1) {
      guess = aux;
    } else if (aux < (int)remaining_answers.size()) {
      guess = remaining_answers[aux];
    } else if (aux == (int)remaining_answers.size()) {
      not_answers = DictMinusRemainingAnswers(remaining_answers);
      guess = not_answers[0];
    } else {
      guess = not_answers[aux - remaining_answers.size()];
    }

    if (depth > 1 && !IsHardModeCompatible(previous_guess, previous_pattern, guess)) {
      continue;
    }

    // A lower bound for the score is the current guess plus next guess minus 1, in case the current guess is among the
    // remaining answers.
    bool among_remaining_answers = depth > 1 ? aux < (int)remaining_answers.size() : set_remaining_answers.count(guess);
    int score = int(remaining_answers.size() + remaining_answers.size()) - int(among_remaining_answers);
    // An upper bound for the score based on the current best score and the parent upper bound score
    int upper_bound_score = min(best_score, parent_score_upper_bound);

    // There are many cases where the current guess splits the answers in buckets of size 1 or doesn't split the answers
    // at all. In both cases, we don't need to do recursive calls to solve the buckets. The cost of bucketing the
    // answers into new vectors is very high. This computes the bucket sizes without actually creating the buckets.
    int largest_bucket_size = 0;
    ++iteration;
    assert(iteration < LLONG_MAX);
    for (short answer : remaining_answers) {
      int pattern = pattern_matrix[guess][answer];
      if (need_reset[pattern] != iteration) {
        bucket_sizes[pattern] = 0;
        need_reset[pattern] = iteration;
      } else {
        ++score;
        if (upper_bound_score <= score) {
          break;
        }
      }
      ++bucket_sizes[pattern];
      largest_bucket_size = max(largest_bucket_size, bucket_sizes[pattern]);
    }

    // Prune: With our updated lower bound score, check if a better guess has been found already.
    if (upper_bound_score <= score) {
      continue;
    }

    // If the remaining answers are all in the same bucket, then the current guess doesn't provide any useful
    // information. This check also prevents an infinite recursion because we would call the recursion to solve the same
    // state as the current one.
    if (largest_bucket_size == (int)remaining_answers.size()) {
      continue;
    }

    // If the guess splits the remaining answers into size-1 buckets, then our current lower bound for the score is the
    // real score.
    if (largest_bucket_size == 1) {
      best_score = min(best_score, score);
      best_guess = score == best_score ? guess : best_guess;
      break;
    }

    // Distribute remaining answers in their corresponding pattern bucket.
    map<int, vector<short>> buckets;
    for (short answer : remaining_answers) {
      int pattern = pattern_matrix[guess][answer];
      if (bucket_sizes[pattern] == 1) continue;
      buckets[pattern].push_back(answer);
    }

    // Speed-up heuristic: process buckets by their size.
    vector<pair<int, vector<short>>> sorted_buckets(buckets.begin(), buckets.end());
    std::sort(sorted_buckets.begin(), sorted_buckets.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second.size() < rhs.second.size(); });

    // Calculate recursively the score for each bucket.
    for (const auto& pattern_and_bucket : sorted_buckets) {
      int pattern = pattern_and_bucket.first;
      const auto& bucket = pattern_and_bucket.second;
      //  Remove the contribution of this bucket to the initial lower bound score calculation.
      score -= int(bucket.size() + bucket.size() - 1);
      // Add the calculated contribution of this bucket to the score.
      const int child_score_upper_bound = upper_bound_score - score;
      score += Dfs(initial_guess, bucket, child_score_upper_bound, depth + 1, guess, pattern);
      // Prune: With the updated score from this bucket, check if a better guess has being found already.
      if (upper_bound_score <= score && pattern_and_bucket != sorted_buckets.back()) {
        score = kInf;
        break;
      }
    }

    best_score = min(best_score, score);
    best_guess = score == best_score ? guess : best_guess;
  }

  if (best_score < kInf) {
    _memo[remaining_answers] = {best_score, best_guess};
    _memo_upper.erase(remaining_answers);
  } else {
    _memo_upper[remaining_answers] = max(_memo_upper[remaining_answers], parent_score_upper_bound);
    memo_upper_size += (int)remaining_answers.capacity();
    if (memo_upper_size > int(5e7)) {
      for (int i = 0; i < kMaxAllowedGuesses; ++i) {
        for (int j = 0; j < dict_size; ++j) {
          for (int k = 0; k <= kMaxEncodedPattern; ++k) {
            memo_upper[i][j][k].clear();
          }
        }
      }
      memo_upper_size = 0;
      dbg("Memo upper cleared.");
    }
  }

  return best_score;
}

int total_score = 0;
int total_answers = 0;
bool csv = false;
bool emoji = true;
void PrintOptimalStrategy(const vector<short>& remaining_answers, const int depth = 1, int previous_guess = -1,
                          int previous_pattern = -1) {
  assert(depth <= kMaxAllowedGuesses);
  if ((int)remaining_answers.size() == 1) {
        cout << dict[remaining_answers[0]] << " ,"[csv] << DecodePattern(kExactMatch, emoji)
         << (emoji ? "" : to_string(depth)) << endl;
    total_score += depth;
    ++total_answers;
    return;
  }

  auto& _memo = memo[depth - 1][previous_guess][previous_pattern];

  int guess;
  int score;

  auto entry = _memo.find(remaining_answers);
  assert(entry != _memo.end());
  tie(score, guess) = entry->second;
  assert(depth == 1 || IsHardModeCompatible(previous_guess, previous_pattern, guess));

  map<int, vector<short>> buckets;
  for (short answer : remaining_answers) {
    int pattern = pattern_matrix[guess][answer];
    buckets[-pattern].push_back(answer);
  }

  cout << dict[guess] << " ,"[csv];
  cout << flush;
  for (const auto& pattern_and_bucket : buckets) {
    int pattern = -pattern_and_bucket.first;
    const vector<short>& bucket = pattern_and_bucket.second;
    if (pattern != -buckets.begin()->first) {
      int multiplier = csv ? 1 + 2 * (depth - 1) : 6 + 13 * (depth - 1);
      cout << string(multiplier, " ,"[csv]);
    }
    cout << DecodePattern(pattern, emoji) << (emoji ? "" : to_string(depth));
    cout << flush;
    if (pattern != kExactMatch) {
      cout << " ,"[csv];
      PrintOptimalStrategy(bucket, depth + 1, guess, pattern);
    } else {
      cout << endl;
      total_score += depth;
      ++total_answers;
    }
  }
}

int main(int argc, char* argv[]) {
  assert(argc == 3 || argc == 4 || argc == 5);
  string game = argv[1];
  string initial_guess = argv[2];
  csv = argc > 3 && string(argv[3]) == "csv" ? true : false;
  emoji = argc > 4 && string(argv[4]) == "emoji" ? true : false;

  ReadGameDict(game);

  SortDictWordsByLetterFrequency();
  ComputePatternMatrix();

  vector<short> remaining_answers;
  int initial_guess_idx = 0;
  for (int i = 0; i < dict_size; ++i) {
    if (initial_guess == dict[i]) {
      initial_guess_idx = i;
    }
    for (int j = 0; j < answers_size; ++j) {
      if (dict[i] == answers[j]) {
        remaining_answers.push_back(short(i));
      }
    }
  }

  int score = Dfs(initial_guess_idx, remaining_answers);
  if (score == kInf) {
    cout << "INITIAL GUESS DOESN'T HAVE AN OPTIMAL STRATEGY THAT SOLVES EVERY WORDLE GAME IN LESS THAN "
         << kMaxAllowedGuesses << " GUESSES.";
    return 0;
  }
  cout.precision(4);
  cout << fixed;
  cout << "[HARD MODE] Strategy for initial guess: " << initial_guess;
  cout << ". Expected average score: " << (double)score / (double)answers_size;
  cout << ". Total guesses used: " << score << "." << endl;
  PrintOptimalStrategy(remaining_answers);
  assert(total_score == score && total_answers == answers_size);
  return 0;
}