head -n 5000 MultiMaCoCu.ca-en.rand.ca > MultiMaCoCu.ca-en.test.ca
head -n 5000 MultiMaCoCu.ca-en.rand.en > MultiMaCoCu.ca-en.test.en
head -n 10000 MultiMaCoCu.ca-en.rand.en | tail -n 5000 > MultiMaCoCu.ca-en.dev.en
head -n 10000 MultiMaCoCu.ca-en.rand.ca | tail -n 5000 > MultiMaCoCu.ca-en.dev.ca
tail -n +10001 MultiMaCoCu.ca-en.rand.ca > MultiMaCoCu.ca-en.train.ca
tail -n +10001 MultiMaCoCu.ca-en.rand.en > MultiMaCoCu.ca-en.train.en