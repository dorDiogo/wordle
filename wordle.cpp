// #define NDEBUG
#include <bits/stdc++.h>
#include <omp.h>

#define dbg(x) cout << ">>> " << x << endl;
#define _ << " | " <<

using namespace std;

const int kNumThreads = 12;
int node_id = 1;
int total_nodes = 1;

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

map<vector<short>, int> memo_exact[kMaxAllowedGuesses];
int memo_exact_size = 0;
map<vector<short>, int> memo_upper[kMaxAllowedGuesses];
int memo_upper_size = 0;

int bucket_sizes[kNumThreads][kMaxEncodedPattern + 1] = {0};
long long need_reset[kNumThreads][kMaxEncodedPattern + 1] = {0};
long long iteration[kNumThreads] = {0};
int best_guess;

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

// Computes the resulting pattern for the corresponding guess/answer pair.
// Encoded as a base-3 number where 0 = GREEN, 1 = YELLOW, 2 = BLACK.
// E.g., ⬛️🟨⬛️🟨🟩 = 2 * 3^4 + 1 * 3^3 + 2 * 3^2 + 1 * 3^1 + 0 * 3^0.
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

pair<bool, int> GetMemoEntry(const map<vector<short>, int>& memo, const vector<short>& remaining_answers) {
  int value = kInf;
  bool found_memo_entry;
#pragma omp critical(memo_access)
  {
    const auto memo_entry = memo.find(remaining_answers);
    found_memo_entry = memo_entry != memo.end();
    if (found_memo_entry) value = memo_entry->second;
  }
  return {found_memo_entry, value};
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
int Dfs(const vector<short>& remaining_answers, const int parent_score_upper_bound = kInf, const int depth = 1,
        int parent_tid = -1) {
  if (depth > kMaxAllowedGuesses) {
    return kInf;
  }
  if ((int)remaining_answers.size() == 1) {
    return 1;
  }
  if (depth + 1 > kMaxAllowedGuesses) {
    return kInf;
  }
  if ((int)remaining_answers.size() == 2) {
    return 3;
  }

  // Score is the sum how many guesses are needed to find the answer for all remaining answers.
  int best_score = kInf;
  bool found_memo_entry;
  tie(found_memo_entry, best_score) = GetMemoEntry(memo_exact[depth - 1], remaining_answers);
  if (found_memo_entry) {
    return best_score;
  }

  auto found_and_value = GetMemoEntry(memo_upper[depth - 1], remaining_answers);
  if (found_and_value.first) {
    if (parent_score_upper_bound <= found_and_value.second) {
      return kInf;
    }
  }

  vector<short> not_answers;
  set<short> set_remaining_answers;
  if (depth == 1) {
    set_remaining_answers = set<short>(remaining_answers.begin(), remaining_answers.end());
  }

#pragma omp parallel for schedule(dynamic) num_threads(kNumThreads) if (depth == 1)
  for (short aux = 0; aux < dict_size; ++aux) {
    if (depth == 1 && (aux + node_id) % total_nodes != 0) {
      continue;
    }
    short guess;

    // Speed-up: process remaining answers first, and then the rest of the dictionary if needed.
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

    // A lower bound for the score is the current guess plus next guess minus 1, in case the current guess is among the
    // remaining answers.
    bool among_remaining_answers = depth > 1 ? aux < (int)remaining_answers.size() : set_remaining_answers.count(guess);
    int score = int(remaining_answers.size() + remaining_answers.size()) - int(among_remaining_answers);
    // An upper bound for the score based on the current best score and the parent upper bound score
    int upper_bound_score = depth > 1 ? min(best_score, parent_score_upper_bound)
                                      : (parent_score_upper_bound == kInf ? best_score : parent_score_upper_bound);

    int tid;
#ifdef _OPENMP
    tid = depth == 1 ? omp_get_thread_num() : parent_tid;
#else
    tid = 0;
#endif

    // There are many cases where the current guess splits the answers in buckets of size 1 or doesn't split the answers
    // at all. In both cases, we don't need to do recursive calls to solve the buckets. The cost of bucketing the
    // answers into new vectors is very high. This computes the bucket sizes without actually creating the buckets.
    int largest_bucket_size = 0;
    ++iteration[tid];
    assert(iteration[tid] < LLONG_MAX);
    for (short answer : remaining_answers) {
      int pattern = pattern_matrix[guess][answer];
      if (need_reset[tid][pattern] != iteration[tid]) {
        bucket_sizes[tid][pattern] = 0;
        need_reset[tid][pattern] = iteration[tid];
      } else {
        ++score;
        if (upper_bound_score <= score) {
          break;
        }
      }
      ++bucket_sizes[tid][pattern];
      largest_bucket_size = max(largest_bucket_size, bucket_sizes[tid][pattern]);
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
      if (depth == 1) {
#pragma omp critical(check_update_score)
        best_score = min(best_score, score);
        best_guess = score == best_score ? guess : best_guess;
      } else {
        best_score = min(best_score, score);
      }
#ifdef _OPENMP
#pragma omp cancel for
      continue;  // in case OMP_CANCELLATION env variable is set to false
#else
      break;
#endif
    }

    // Distribute remaining answers in their corresponding pattern bucket.
    map<int, vector<short>> buckets;
    for (short answer : remaining_answers) {
      int pattern = pattern_matrix[guess][answer];
      if (bucket_sizes[tid][pattern] == 1) continue;
      buckets[pattern].push_back(answer);
    }

    // Speed-up heuristic: process buckets by their size.
    vector<pair<int, vector<short>>> sorted_buckets(buckets.begin(), buckets.end());
    std::sort(sorted_buckets.begin(), sorted_buckets.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second.size() < rhs.second.size(); });

    // Calculate recursively the score for each bucket.
    for (const auto& pattern_and_bucket : sorted_buckets) {
      const auto& bucket = pattern_and_bucket.second;
      // Remove the contribution of this bucket to the initial lower bound score calculation.
      score -= int(bucket.size() + bucket.size() - 1);
      // Add the calculated contribution of this bucket to the score.
      const int child_score_upper_bound = upper_bound_score - score;
      score += Dfs(bucket, child_score_upper_bound, depth + 1, tid);
      // Prune: With the updated score from this bucket, check if a better guess has being found already.
      if (upper_bound_score <= score && pattern_and_bucket != sorted_buckets.back()) {
        score = kInf;
        break;
      }
    }

    if (depth == 1) {
#pragma omp critical(check_update_score)
      {
        best_score = min(best_score, score);
        best_guess = score == best_score ? guess : best_guess;
        std::time_t clock_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        string clock_time_string = ctime(&clock_time);
        clock_time_string.pop_back();
        dbg(clock_time_string _ guess _ dict[guess] _(score < kInf ? to_string(score) : "pruned"));
      }
    } else {
      best_score = min(best_score, score);
    }
  }

#pragma omp critical(memo_access)
  {
    if (best_score < kInf) {
      memo_exact[depth - 1][remaining_answers] = best_score;
      memo_upper[depth - 1].erase(remaining_answers);
      memo_exact_size += (int)remaining_answers.capacity();
      if (memo_exact_size > int(15e7)) {
        for (int i = 0; i < kMaxAllowedGuesses; ++i) {
          memo_exact[i].clear();
        }
        memo_exact_size = 0;
        dbg("Memo exact cleared.");
      } else {
      }
    } else {
      memo_upper[depth - 1][remaining_answers] =
          max(memo_upper[depth - 1][remaining_answers], parent_score_upper_bound);
      memo_upper_size += (int)remaining_answers.capacity();
      if (memo_upper_size > int(15e7)) {
        for (int i = 0; i < kMaxAllowedGuesses; ++i) {
          memo_upper[i].clear();
        }
        memo_upper_size = 0;
        dbg("Memo upper cleared.");
      }
    }
  }
  return best_score;
}

int main(int argc, char* argv[]) {
#ifdef _OPENMP
  assert(getenv("OMP_CANCELLATION") != NULL);
#endif
  assert(argc == 2 || argc == 4 || argc == 5);

  string game_dict_file = argv[1];

  // When running this code in multiple nodes.
  if (argc >= 4) {
    node_id = atoi(argv[2]);
    total_nodes = atoi(argv[3]);
  }

  // Know all words that score lower than this value.
  int score_upper_bound = argc == 5 ? atoi(argv[4]) : kInf;

  ReadGameDict(game_dict_file);
  
  // Speed-up: process guesses by words with letters that are more frequent in the dictionary.
  SortDictWordsByLetterFrequency();
  ComputePatternMatrix();

  vector<short> remaining_answers;
  for (int i = 0; i < dict_size; ++i) {
    for (int j = 0; j < answers_size; ++j) {
      if (dict[i] == answers[j]) {
        remaining_answers.push_back(short(i));
      }
    }
  }
  assert((int)remaining_answers.size() == answers_size);
  int best_score = Dfs(remaining_answers, score_upper_bound);
  dbg("BEST" _ best_guess _ dict[best_guess] _ best_score _ (double)best_score / (double)answers_size);
  return 0;
}