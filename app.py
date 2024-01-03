import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.header("Binomial Distribution of Coin Flips")

sequence_length = st.number_input("Total number of flips in a sequence", min_value=1, value=10)
probability_heads = st.number_input("Probability of heads (is this a weighted coin?)", min_value=0.0, max_value=1.0, value=0.5)
number_sequences = st.number_input("Number of sequences you want", min_value=1, value=1000)
st.divider()

st.subheader("Graphing the number of heads per sequence")

st.write("$$$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{1}{2}(\\frac{x-\mu}{\sigma})^2} $$$ = Normal Distribution Formula")
st.write("$$$ n $$$ = %d = Total number of events in a sequence" % sequence_length)
st.write("$$$ p $$$ = %f = Probability of getting the desired event ONCE" % probability_heads)
st.write("$$$ \mu $$$ = Mean")
st.write("$$$ \sigma $$$ = Standard Deviation")
st.write("$$$ \sigma^2 $$$ = Variance")
st.write("$$$ \sigma = n * p * (1 - p) $$$")
st.write("$$$ \mu = n * p $$$")
st.write("$$$ \sigma = %d * %f * (1 - %f) = %f $$$" % (sequence_length, probability_heads, probability_heads, sequence_length * probability_heads * (1 - probability_heads)))
st.write("$$$ \mu = %d * %f = %f $$$" % (sequence_length, probability_heads, sequence_length * probability_heads))
st.write("$$$ f(x) = \\frac{1}{%f\\sqrt{2\\pi}}e^{-\\frac{1}{2}(\\frac{x-%f}{%f})^2} $$$" % (sequence_length * probability_heads * (1 - probability_heads), sequence_length * probability_heads, sequence_length * probability_heads * (1 - probability_heads)))

mean = sequence_length * probability_heads
variance = sequence_length * probability_heads * (1 - probability_heads)
std_dev = np.sqrt(variance)
f = lambda x: (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / (std_dev)) ** 2)

flips = np.random.binomial(sequence_length, probability_heads, number_sequences)

if st.button('Rerun', use_container_width=True):
    st.rerun()

fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# n flips histogram
ax.hist(flips, bins=range(sequence_length+2), align='left', rwidth=0.8, color='blue', edgecolor='black', density=False)
ax.set_xlabel('Number of successes (heads) in one sequence')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Binomial Distribution for %d Coin Flips' % sequence_length)

# normal distribution
bin_heights, bin_borders = np.histogram(flips, bins=range(sequence_length+2), density=False)
bin_width = np.diff(bin_borders)
scaling_factor = number_sequences * bin_width[0]
x = np.linspace(0, sequence_length, 1000)
y_values = f(x) * scaling_factor
ax.plot(x, y_values, color='red', label='Normal Distribution')

ax2.boxplot(flips, positions=[sequence_length*probability_heads], vert=False, widths=bin_width[0]*0.5, patch_artist=True)
ax2.set_xlim(ax.get_xlim())
ax2.get_yaxis().set_visible(False)

st.pyplot(fig)

df = pd.DataFrame([
    ['Variance', variance],
    ['Standard Deviation', std_dev],
    ['Mean', mean],
    ['Mode', np.argmax(bin_heights)],
    ['Median', np.median(flips)],
    ['Range', np.max(flips) - np.min(flips)],
    ['Maximum', np.max(flips)],
    ['Minimum', np.min(flips)]
], columns=['Statistic', 'Value'])
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Calculate the probability of getting a certain number of heads")
st.write("$$$ P(x) = \\frac{n!}{(n-k)!k!}*p^k(1-p)^{n-k} $$$ = Binomial Distribution Formula")
k = st.number_input("Number of heads you want", min_value=0, max_value=sequence_length, value=5)
st.write("$$$ k $$$ = %d = Number of desired events within a sequence" % k)
st.write("$$$ n $$$ = %d = Total number of events in a sequence" % sequence_length)
st.write("$$$ p $$$ = %f = Probability of getting the desired event ONCE" % probability_heads)
st.write('$$$ P(x) $$$ = Probability of getting the desired number of heads in a single sequence')
st.write("$$$ P(x) = \\frac{%d!}{(%d-%d)!%d!}*%f^%d(1-%f)^{%d-%d} $$$" % (sequence_length, sequence_length, k, k, probability_heads, k, probability_heads, sequence_length, k))
desired_probability = (np.math.factorial(sequence_length) / (np.math.factorial(sequence_length-k) * np.math.factorial(k))) * (probability_heads**k) * ((1-probability_heads)**(sequence_length-k))
st.write("$$$ P(x) = %f $$$ = The probability of getting your desired number of heads in a single sequence" % desired_probability)