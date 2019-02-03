
echo $1
echo $2
gnuplot -e "
	unset key;
	set datafile separator ',';
	set terminal pngcairo enhanced size 960,720;
	set yrange [0:0.35];
	plot '$1' using 0:1  with line ; 
	replot '$1' using 0:2 with line; 
	replot '$1' using 0:3 with line;
	replot '$1' using 0:4 with line;
	replot '$1' using 0:5 with line;
	set output '$2';
	replot;
	quit
	" > /dev/null
