-- Copyright 2018 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

import "regent"

local c = regentlib.c

-- A standard Monte Carlo simulation to estimate the area of a circle of radius 1.
--
-- Picture a circle of radius 1 inscribed inside a square.  Now throw N darts at the square, hitting
-- random locations (all darts hit somewhere in the square).  If T of the darts land inside the circle,
-- then the circle's fraction of the square's area is about T/N.  Since the area of the square is 4
-- (a circle of radius 1 incribes inside a 2x2 square), (T/N) * 4 is our estimate of the area of the circle.
--
-- The simulation carries out this simulation on 1/4 of the square. Imagine both the square and the
-- circle have their center at the origin of the (x,y) plane.  Now choose a random x in the range 0..1 and
-- a random y in the range 0..1.  If x^2 + y^2 <= 1, then this "dart" lands inside the upper right quadrant
-- of the circle, otherwise it lands somewhere outside the circle in the upper right quadrant of the square.
--

--
-- TODO  Modify the code to make 2500 trials at a time in each of 4 parallel tasks.
--
task hits(iterations : int64)
  var total: int64 = 0
  for i = 1, iterations do
    var x : double = c.drand48()
    var y : double = c.drand48()
    if (x * x) + (y * y) <= 1.0 then
      total = total +1
    end
  end
  return total
end

task main()
  var iterations : int64 = 2500

  var hits1 = hits(iterations)
  var hits2 = hits(iterations)
  var hits3 = hits(iterations)
  var hits4 = hits(iterations)  
  var totalhits = hits1 + hits2 + hits3 + hits4
  c.printf("The area of a unit circle is approximately: %5.4f\n", totalhits / [double](iterations))
end

regentlib.start(main)
