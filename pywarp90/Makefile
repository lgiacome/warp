# Possible targets: (install is the default)
# install, debug, clean, cleandebug: for serial, python2
# pinstall, pdebug, pclean, pcleandebug: for parallel, python2
# install3, debug3, clean3, clean3debug: for serial, python3
# pinstall3, pdebug3, pclean3, pclean3debug: for parallel, python3
# installall, debugall, cleanall

install:
	make -f Makefile.Forthon install

pinstall:
	make -f Makefile.Forthon.pympi install

install3:
	make -f Makefile.Forthon3 install

pinstall3:
	make -f Makefile.Forthon3.pympi install

debug:
	make -f Makefile.Forthon DEBUG='-g --farg "-O0"' BUILDBASEDIR=builddebug install

pdebug:
	make -f Makefile.Forthon.pympi DEBUG='-g --farg "-O0"' BUILDBASEDIR=buildparalleldebug install

debug3:
	make -f Makefile.Forthon3 DEBUG='-g --farg "-O0"' BUILDBASEDIR=build3debug install

pdebug3:
	make -f Makefile.Forthon3.pympi DEBUG='-g --farg "-O0"' BUILDBASEDIR=build3paralleldebug install

clean:
	make -f Makefile.Forthon clean

pclean:
	make -f Makefile.Forthon.pympi clean

clean3:
	make -f Makefile.Forthon3 clean

pclean3:
	make -f Makefile.Forthon3.pympi clean

cleandebug:
	make -f Makefile.Forthon BUILDBASEDIR=builddebug clean

pcleandebug:
	make -f Makefile.Forthon.pympi BUILDBASEDIR=buildparalleldebug clean

clean3debug:
	make -f Makefile.Forthon3 BUILDBASEDIR=build3debug clean

pclean3debug:
	make -f Makefile.Forthon3.pympi BUILDBASEDIR=build3paralleldebug clean

installall: install pinstall install3 pinstall3

debugall: debug pdebug debug3 pdebug3

cleanall: clean pclean clean3 pclean3 cleandebug pcleandebug clean3debug pclean3debug
	rm -rf build*

help:
	@echo "Possible targets: (install is the default)"
	@echo "install, debug, clean, cleandebug: for serial, python2"
	@echo "pinstall, pdebug, pclean, pcleandebug: for parallel, python2"
	@echo "install3, debug3, clean3, clean3debug: for serial, python3"
	@echo "pinstall3, pdebug3, pclean3, pclean3debug: for parallel, python3"
	@echo "installall, debugall, cleanall"
	@echo "help"

