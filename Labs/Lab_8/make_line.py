def make_line(x, y, alpha):
  # Makes a line through two input points and evalulates it at point alpha
  f_alpha = ((y[0]-y[1])/(x[0]-x[1]))*alpha
  return f_alpha
