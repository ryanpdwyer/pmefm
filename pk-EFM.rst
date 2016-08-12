pk-EFM
======

The phasekick electric force microscopy workup needs to be cleaned up, simpified. The workup has a few distinct stages.



1. Raw h5 file, cantilever oscillation data => cantilever frequency, phase, amplitude + important time points (t1, t2, tp, t3)
     - Note: Should be general: tp_light, tp_tip 
     - Currently don't have a good representation of this intermediate state, which is why everything is too coupled together.
     - Print report from here
2. Python freq, amp, phase information => dphi
   Python freq, amp, phase information => phase_correction
   Etc (At this point, relevant information summarized in a dataframe).
   - Do I need a supplemental dictionary?
3. csv file(s) (+ dictionary?) => fits (simple best fit in python, better best fit in pystan)
    - Print report of best fit workup
