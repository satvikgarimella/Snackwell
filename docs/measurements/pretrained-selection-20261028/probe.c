/* CPU-only quality probe around the author's unmodified llama2.c reference.
 * run.c and its MIT license are downloaded together; hashes are in sources.json.
 * Count new tokens separately from prompt tokens and retain the entire output.
 */
#define TESTING
#include "run.c"

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    Transformer model;
    Tokenizer tokenizer;
    Sampler sampler;
    build_transformer(&model, argv[1]);
    build_tokenizer(&tokenizer, argv[2], model.config.vocab_size);
    build_sampler(&sampler, model.config.vocab_size, atof(argv[4]), 1.0f,
                  strtoull(argv[5], NULL, 10));
    int *ids = malloc((strlen(argv[3]) + 3) * sizeof(int));
    int count = 0;
    encode(&tokenizer, argv[3], 1, 0, ids, &count);
    int limit = atoi(argv[6]);
    if (count < 1 || limit < 1 || count + limit > model.config.seq_len) return 3;
    float *logits = NULL;
    for (int pos = 0; pos < count; pos++) logits = forward(&model, ids[pos], pos);
    int previous = ids[count - 1], ended = 0, emitted = 0;
    for (int i = 0; i < limit; i++) {
        int token = sample(&sampler, logits);
        emitted++;
        if (token == 1 || token == 2) { ended = 1; break; }
        safe_printf(decode(&tokenizer, previous, token));
        previous = token;
        if (i + 1 < limit) logits = forward(&model, token, count + i);
    }
    fprintf(stderr, "{\"prompt_tokens\":%d,\"new_tokens\":%d,\"ended\":%d}\n",
            count, emitted, ended);
    free(ids);
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&model);
    return 0;
}
