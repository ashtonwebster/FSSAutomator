hash = {};

lines = $stdin.readlines
lines.each() { |line|
    hash[line.chomp.split("-").sort!] = 1;
}

hash.keys.each { |key|
    count = 1
    key.each { |x| 
	   print "#{x}"
	   if (count != key.length) then
		  print "-"
	   end
	   count += 1
    }
    puts
}
