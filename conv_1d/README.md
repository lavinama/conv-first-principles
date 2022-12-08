There are two different types of convolutions for 1D:

<p align="center">
  <img src="./../media/types_conv_1d.png", width=400 />
</p>

**input:** [$W$], **filter:** [$k$]
### Full
Flips the filter before “sliding” the two across one another until every point has been passed by all the filter values.
**output:** [W + k - 1]

```python
np.convolve([1, 2, 3, 4], [0.5, 1])
>>> [0.5 2.  3.5 5.  4. ]
```

### Same
Flips the filter before “sliding” the two across one another until we reach the final input value.
**output:** [max(W, k)]

```python
np.convolve([1, 2, 3, 4], [0.5, 1], 'same')
>>> [0.5 2.  3.5 5. ]
```

### Valid
Flips the filter before “sliding” the two across one another starting from the first input and until the we reach the final value of the input.
**output:** [max(W, k) - min(W, k) + 1]

```python
np.convolve([1, 2, 3, 4], [0.5, 1], 'valid')
>>> [2.  3.5 5. ]
```